#!/usr/bin/env bash
# sbatch_guard.sh — the ONLY way this project submits jobs (CLAUDE.md "Budget guard" and rules).
#
#   scripts/sbatch_guard.sh [sbatch options...] <script.sbatch|script.sh> [script args...]
#
# Runs ON THE CLUSTER (invoked through scripts/remote.sh). It
#   1. reads config/budget.yaml,
#   2. computes SUs this project has consumed so far from sacct (billing TRES x elapsed MINUTES);
#      "this project" = jobs whose name starts with budget.yaml's job_name_prefix (every template
#      is named "<project>-..."); without a prefix it is every job of $USER on the account,
#   3. adds the still-reserved SUs of every job of ours that is PENDING or RUNNING (rate x the
#      minutes left of its time limit) — a queued job's whole reservation counts against the cap,
#      and a pending array range counts once per task,
#   4. estimates the SUs the new job could consume at its full time limit (from #SBATCH directives
#      plus any command-line overrides, which win): x the task count of an --array (a %N throttle
#      limits how many tasks run at once, not how many run), and a whole node for --exclusive,
#   5. refuses when used + reserved + projected would exceed max_project_sus, when active jobs +
#      the new job's tasks would exceed max_concurrent_jobs (each array task is a job), when the
#      job asks for more GPU nodes than max_gpu_nodes_per_job, when its time limit exceeds
#      max_job_hours, when its name lacks job_name_prefix (the ledger would not count it), or when
#      its --output or --error directory does not exist and cannot be created (the job would die with no log),
#      when it would run in a read-only submit directory (a code snapshot, scripts/snapshot.py) with a missing
#      or relative --output, a relative --error or a --chdir elsewhere, or when the ledger is not writable,
#   6. otherwise calls sbatch and appends a line to the ledger: $GUARD_LOG, by default $REMOTE_KEEP/submissions.log
#      (kit.env), else jobs/submissions.log (git-ignored).
# It FAILS CLOSED: if sacct or squeue errors, or a spec cannot be parsed, nothing is submitted.
#
# SU ACCOUNTING — verified 2026-10-25 against CARC docs and the cluster itself (LESSONS #1-2):
#   CARC "Running Jobs" guide (https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs;
#   GPU table also at https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming;
#   other CARC URLs for this, e.g. .../discovery/running-jobs, 404): SUs are charged per MINUTE of RESERVED resources:
#       1 CPU = 1 SU/min, 4 GB memory = 1 SU/min, 1 L40S/A100/A40 GPU = 8 SU/min, 1 V100/P100 = 4 SU/min.
#       Worked examples: "1 CPU with 8 GB memory used for 1 hour = 180 SUs";
#       "8 CPUs with 320 GB memory and 8 A100 GPUs used for 1 hour = 9120 SUs".
#   This is exactly SLURM's `billing` TRES (gpu partition TRESBillingWeights=CPU=1.0,Mem=0.25G,
#   GRES/gpu:{l40s,a100,a40}=8.0,GRES/gpu:{v100,p100}=4.0) times minutes.  So:
#       SUs(job) = billing x ElapsedRaw / 60
#   Full A100 node (64 CPU, 256 GB, 2xA100) = 64 + 64 + 16 = 144 SU/min = 8,640 SU/hour.
#   For every job we read `billing=` from sacct's TRES string; we also recompute it from
#   cpu/mem/gres with the weights above and take the LARGER (conservative).
#   Unknown GPU types are charged at 8. sacct is the only ledger: myaccount drifts between calls,
#   sshare RawUsage is half-life decayed, sreport lags by hours (LESSONS "accounting").
set -euo pipefail

GUARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUDGET="${BUDGET_FILE:-$GUARD_DIR/config/budget.yaml}"
# The ledger lives in $REMOTE_KEEP (kit.env): outside the --delete mirror and the read-only code snapshots.
_KEEP=$( { sed -n 's/^REMOTE_KEEP=//p' "$GUARD_DIR/kit.env" | head -1; } 2>/dev/null || true)
LOG="${GUARD_LOG:-${_KEEP:+$_KEEP/submissions.log}}"; LOG="${LOG:-$GUARD_DIR/jobs/submissions.log}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"          # tests override these with stubs
SACCT_BIN="${SACCT_BIN:-sacct}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
PY="${PYTHON_BIN:-python3}"

yaml_get() {  # yaml_get <key> — flat "key: value" lookup; strips comments and quotes
  awk -F': *' -v k="$1" '$1==k {v=$2; sub(/ *#.*/,"",v); gsub(/"/,"",v); print v; exit}' "$BUDGET"
}
MAX_SUS=$(yaml_get max_project_sus)
MAX_JOBS=$(yaml_get max_concurrent_jobs)
MAX_GPU_NODES=$(yaml_get max_gpu_nodes_per_job)
MAX_HOURS=$(yaml_get max_job_hours)
WARN_FRAC=$(yaml_get warn_at_fraction)
START=$(yaml_get project_start_date)
ACCOUNT=$(yaml_get account)
PREFIX=$(yaml_get job_name_prefix)   # optional; empty = count every job of $USER on the account
TARGET=$(yaml_get target_sus)        # optional soft target: going past it needs PAST_TARGET_OK=<date of Shresth's OK>
APPROVALS=$(yaml_get approvals_file) # optional job-type registry, relative to the repo (CLAUDE.md §4, §11 rule 6)
for v in MAX_SUS MAX_JOBS MAX_GPU_NODES MAX_HOURS WARN_FRAC START ACCOUNT; do
  [ -n "${!v}" ] || { echo "sbatch_guard: $v missing from $BUDGET" >&2; exit 2; }
done

die() { echo "sbatch_guard: REFUSED — $*" >&2; exit 1; }
# exit 4 = a human decision is needed: nothing was submitted; ask Shresth in chat, then `make approve`
needs_approval() { echo "sbatch_guard: NEEDS_APPROVAL $*" >&2; exit 4; }

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/sbatch_guard.XXXXXX")
trap 'rm -rf "$TMPD"' EXIT

# One Python helper for all accounting. Modes:
#   used [prefix]   stdin: "TRES|ElapsedRaw|JobName" rows -> SU consumed (sum of rate x minutes)
#   reserved <h>    stdin: "JobID|State|ReqTRES|AllocTRES|TimelimitRaw|ElapsedRaw" rows -> SU still reserved
#   active [prefix] stdin: squeue "JobID|JobName" rows -> "<jobs, array tasks expanded> <base ids,csv|->"
#   tasks <spec>    --array spec                       -> number of tasks
#   hours <t>       SLURM time spec                    -> hours (inf for UNLIMITED/INFINITE)
#   gb <m>          SLURM memory spec (default unit MB)-> GB
SU_PY='
import math, re, sys
W = {"a100": 8, "a40": 8, "l40s": 8, "v100": 4, "p100": 4}
def rate(tres):
    f = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
    billing = float(f.get("billing") or 0)
    cpu = float(f.get("cpu") or 0)
    mem_gb = 0.0
    if f.get("mem"):
        m = re.match(r"([\d.]+)([KMGT]?)", f["mem"])
        mem_gb = float(m.group(1)) * {"K": 1/1048576, "M": 1/1024, "G": 1, "T": 1024, "": 1/1048576}[m.group(2)]
    typed = {k.split(":", 1)[1].lower(): float(v) for k, v in f.items() if k.startswith("gres/gpu:")}
    gpu = sum(n * W.get(t, 8) for t, n in typed.items())
    if not typed and f.get("gres/gpu"):
        gpu = float(f["gres/gpu"]) * 8
    return max(billing, cpu + mem_gb * 0.25 + gpu)
def hours(t):
    t = t.strip()
    if t.upper() in ("UNLIMITED", "INFINITE", "PARTITION_LIMIT"):
        return math.inf
    if "-" in t:                                   # days-hours[:minutes[:seconds]]
        d, rest = t.split("-", 1)
        p = [int(x) for x in rest.split(":")] if rest else [0]
        h, m, s = (p + [0, 0])[:3]
        return int(d) * 24 + h + m / 60 + s / 3600
    p = [int(x) for x in t.split(":")]
    if len(p) == 1: return p[0] / 60               # minutes
    if len(p) == 2: return p[0] / 60 + p[1] / 3600 # minutes:seconds
    return p[0] + p[1] / 60 + p[2] / 3600          # hours:minutes:seconds
def gb(m):
    mm = re.fullmatch(r"([\d.]+)([KMGTkmgt]?)", m.strip())
    if not mm: raise SystemExit(f"bad memory spec {m!r}")
    return float(mm.group(1)) * {"K": 1/1048576, "M": 1/1024, "G": 1, "T": 1024, "": 1/1024}[mm.group(2).upper()]
def ntasks(spec):
    # "0-9", "1,3,5-7", "0-15:4"; the %N throttle caps tasks running at once, not tasks run
    n = 0
    for part in spec.split("%", 1)[0].split(","):
        part = part.strip()
        if not part:
            continue
        rng, _, step = part.partition(":")
        a, _, b = rng.partition("-")
        lo, hi, st = int(a), int(b or a), int(step or 1)
        if st < 1 or hi < lo:
            raise SystemExit(f"bad array range {part!r}")
        n += len(range(lo, hi + 1, st))
    if n < 1:
        raise SystemExit(f"empty array spec {spec!r}")
    return n
def id_tasks(jid):
    # 123 or 123_4 -> 1; a pending array range 123_[5-99%8] -> 95
    m = re.search(r"_\[([^\]]*)\]", jid)
    return ntasks(m.group(1)) if m else 1
mode = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 and mode in ("used", "active") else ""
if mode == "used":
    tot = 0.0
    for line in sys.stdin:
        parts = line.rstrip("\n").split("|", 2)
        if len(parts) < 2 or not parts[0]:
            continue
        if prefix and not (len(parts) > 2 and parts[2].startswith(prefix)):
            continue
        tot += rate(parts[0]) * int(parts[1] or 0) / 60
    print(f"{tot:.1f}")
elif mode == "reserved":
    tot = 0.0
    max_min = float(sys.argv[2]) * 60
    for line in sys.stdin:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 6 or not parts[0] or "." in parts[0]:
            continue
        _, state, req, alloc, limit, elapsed = parts[:6]
        if not (state.startswith("PENDING") or state.startswith("RUNNING") or state.startswith("REQUEUED")
                or state.startswith("SUSPENDED") or state.startswith("CONFIGURING")):
            continue
        tres = alloc if alloc else req
        lim = float(limit) if limit.isdigit() else max_min       # unknown limit: assume the maximum we allow
        left = max(0.0, lim - int(elapsed or 0) / 60)
        tot += rate(tres) * left * id_tasks(parts[0])
    print(f"{tot:.1f}")
elif mode == "active":
    n, ids = 0, []
    for line in sys.stdin:
        jid, _, name = line.strip().partition("|")
        m = re.match(r"\d+", jid)
        if not m or (prefix and not name.startswith(prefix)):
            continue
        n += id_tasks(jid)
        if m.group(0) not in ids:
            ids.append(m.group(0))
    print(n, ",".join(ids) or "-")
elif mode == "tasks":
    print(ntasks(sys.argv[2]))
elif mode == "hours":
    h = hours(sys.argv[2]); print("inf" if math.isinf(h) else f"{h:.4f}")
elif mode == "gb":
    print(f"{gb(sys.argv[2]):.4f}")
'
py_su() { "$PY" -c "$SU_PY" "$@"; }

# ---------- 1. SUs used so far (billing-minutes of every job since the project start) ----------
used_sus() {
  "$SACCT_BIN" -u "$USER" -A "$ACCOUNT" -S "$START" -X -n -P --format=AllocTRES,ElapsedRaw,JobName > "$TMPD/used.txt" \
    || die "sacct failed (used SUs); refusing to submit without accounting"
  py_su used "$PREFIX" < "$TMPD/used.txt" || die "cannot parse sacct output (used SUs)"
}

# ---------- 2. our active jobs (array tasks expanded), and the SUs they still hold reserved ----------
"$SQUEUE_BIN" -u "$USER" -h -o '%i|%j' > "$TMPD/active.txt" || die "squeue failed; refusing to submit without knowing active jobs"
ACT=$(py_su active "$PREFIX" < "$TMPD/active.txt") || die "cannot parse squeue output (active jobs)"
ACTIVE=${ACT%% *}; ACTIVE_IDS=${ACT#* }; [ "$ACTIVE_IDS" != "-" ] || ACTIVE_IDS=""
reserved_sus() {
  [ -n "$ACTIVE_IDS" ] || { echo 0.0; return; }
  "$SACCT_BIN" -j "$ACTIVE_IDS" -X -n -P --format=JobID,State,ReqTRES,AllocTRES,TimelimitRaw,ElapsedRaw > "$TMPD/reserved.txt" \
    || die "sacct failed (active reservations); refusing to submit without accounting"
  py_su reserved "$MAX_HOURS" < "$TMPD/reserved.txt" || die "cannot parse sacct output (active reservations)"
}

# ---------- 3. parse the request: #SBATCH directives, then CLI overrides ----------
# Options that take a separate value (sbatch's documented ones we might plausibly see). Any other
# non-option word before the script is treated as the previous option's value, so an unknown
# value-taking option can never be mistaken for the job script.
VALUE_OPTS=" -N -n -c -t -p -A -J -o -e -d -G -C -D -a -w -x -q -L -M -i -b -F --nodes --ntasks --cpus-per-task --time \
 --partition --account --gres --gpus --gpus-per-task --gpus-per-node --gpus-per-socket --cpus-per-gpu --mem --mem-per-cpu \
 --mem-per-gpu --constraint --dependency --ntasks-per-node --ntasks-per-core --ntasks-per-socket --ntasks-per-gpu --output \
 --error --input --job-name --chdir --export --array --mail-type --mail-user --qos --nodelist --exclude --begin --signal \
 --licenses --reservation --time-min --comment --wckey --gpu-bind --open-mode --cluster-constraint --distribution "
is_script() { case "$1" in *.sbatch|*.sh) [ -f "$1" ];; *) return 1;; esac; }
OPTS=(); SCRIPT=""; SCRIPT_ARGS=()
while [ $# -gt 0 ]; do
  if [ -z "$SCRIPT" ]; then
    if [ "${1#-}" != "$1" ]; then
      OPTS+=("$1")
      if [ "${1#*=}" = "$1" ] && [ "${VALUE_OPTS#* $1 }" != "$VALUE_OPTS" ]; then
        [ $# -ge 2 ] || die "option $1 needs a value"
        shift; OPTS+=("$1")
      fi
    elif is_script "$1"; then SCRIPT="$1"
    elif [ ${#OPTS[@]} -gt 0 ]; then OPTS+=("$1")      # value of an option we do not know
    else die "'$1' is not a job script (*.sbatch / *.sh regular file)"
    fi
  else SCRIPT_ARGS+=("$1"); fi
  shift
done
[ -n "$SCRIPT" ] || die "no job script (*.sbatch / *.sh) given"
case " ${OPTS[*]-} " in *" --wrap"*) die "--wrap jobs are not allowed (cannot be costed)";; esac

all_opts() {  # directives first, then CLI (last wins)
  grep -E '^#SBATCH' "$SCRIPT" | sed 's/^#SBATCH *//' | tr -s ' ' '\n'
  printf '%s\n' ${OPTS[@]+"${OPTS[@]}"}
}
getopt_val() {  # getopt_val <long> [<short>] -> last value seen, supports "--k v", "--k=v", "-k v", "-kv"
  local long="$1" short="${2:-}"
  all_opts | awk -v L="--$long" -v S="${short:+-$short}" '
    { if ($0==L || (S!="" && $0==S)) { getline v; print v }
      else if (index($0, L"=")==1) { print substr($0, length(L)+2) }
      else if (S!="" && index($0,S)==1 && length($0)>length(S) && substr($0,1,2)!="--") { print substr($0, length(S)+1) } }' | tail -1
}
NODES=$(getopt_val nodes N);          NODES=${NODES:-1}; NODES=${NODES##*-}     # MIN-MAX ranges cost at MAX
case "$NODES" in ''|*[!0-9]*) die "cannot parse --nodes '$NODES'";; esac
PART=$(getopt_val partition p);       PART=${PART:-main}
TIME=$(getopt_val time t);            TIME=${TIME:-01:00:00}
CPUS=$(getopt_val cpus-per-task c)
NTASKS=$(getopt_val ntasks n);        NTASKS=${NTASKS:-1}
NTPN=$(getopt_val ntasks-per-node)
GRES=$(getopt_val gres)
GPT=$(getopt_val gpus-per-task)
GPN=$(getopt_val gpus-per-node)
GPUS=$(getopt_val gpus G)
CONSTRAINT=$(getopt_val constraint C)
MEM=$(getopt_val mem)
MPC=$(getopt_val mem-per-cpu)
MPG=$(getopt_val mem-per-gpu)
ARRAY=$(getopt_val array a)
JOBNAME=$(getopt_val job-name J)
OUTPUT=$(getopt_val output o)
ERROR=$(getopt_val error e)
EXCLUSIVE=$(all_opts | grep -cE '^--exclusive(=.*)?$' || true)
TASKS=1
if [ -n "$ARRAY" ]; then TASKS=$(py_su tasks "$ARRAY" 2>/dev/null) || die "cannot parse --array '$ARRAY'"; fi

HOURS=$(py_su hours "$TIME") || die "cannot parse --time '$TIME'"
[ "$HOURS" != "inf" ] || die "time limit ${TIME} is unlimited"

# GPUs per node and type from whichever GPU option was used
gpu_spec() {  # "<type>:<n>" or "<n>" -> "type n"
  local s="$1" t="" n=""
  if [ "${s#*:}" != "$s" ]; then t="${s%%:*}"; n="${s##*:}"; else n="$s"; fi
  echo "${t:-a100} ${n:-1}"
}
TASKS_PER_NODE=${NTPN:-$(( (NTASKS + NODES - 1) / NODES ))}
GPU_TYPE=a100; GPUS_PER_NODE=0
if   [ -n "$GRES" ]; then set -- $(gpu_spec "${GRES#gpu:}"); GPU_TYPE=$1; GPUS_PER_NODE=$2
elif [ -n "$GPN" ];  then set -- $(gpu_spec "$GPN");         GPU_TYPE=$1; GPUS_PER_NODE=$2
elif [ -n "$GPT" ];  then set -- $(gpu_spec "$GPT");         GPU_TYPE=$1; GPUS_PER_NODE=$(( $2 * TASKS_PER_NODE ))
elif [ -n "$GPUS" ]; then set -- $(gpu_spec "$GPUS");        GPU_TYPE=$1; GPUS_PER_NODE=$(( ($2 + NODES - 1) / NODES ))
fi
case "$(echo "$GPU_TYPE" | tr 'A-Z' 'a-z')" in v100|p100) GW=4;; *) GW=8;; esac

# CPUs per node: explicit, else all 64 on a GPU node (exclusive assumption, conservative), else tasks
if [ -n "$CPUS" ]; then CPUS_PER_NODE=$(( CPUS * TASKS_PER_NODE ))
elif [ "$GPUS_PER_NODE" -gt 0 ]; then CPUS_PER_NODE=64
else CPUS_PER_NODE=$TASKS_PER_NODE; fi
[ "$CPUS_PER_NODE" -lt 1 ] && CPUS_PER_NODE=1

# Memory per node. --mem=0 (any unit) = the whole node: 768 GB on L40S nodes, 1.5 TB on largemem,
# 256 GB otherwise (the largest node memory on the other partitions we use). Default 2 GB/CPU
# (DefMemPerCPU=2048 on Discovery).
node_mem_gb() {
  case "$(echo "$PART $GRES $CONSTRAINT $GPU_TYPE" | tr 'A-Z' 'a-z')" in
    *largemem*) echo 1536;; *l40s*) echo 768;; *) echo 256;;
  esac
}
if [ -n "$MEM" ]; then
  MEM_GB=$(py_su gb "$MEM") || die "cannot parse --mem '$MEM'"
  "$PY" -c "import sys; sys.exit(0 if float('$MEM_GB') == 0 else 1)" && MEM_GB=$(node_mem_gb)
elif [ -n "$MPC" ]; then
  MEM_GB=$("$PY" -c "print($(py_su gb "$MPC") * $CPUS_PER_NODE)") || die "cannot parse --mem-per-cpu '$MPC'"
elif [ -n "$MPG" ]; then
  MEM_GB=$("$PY" -c "print($(py_su gb "$MPG") * max($GPUS_PER_NODE, 1))") || die "cannot parse --mem-per-gpu '$MPG'"
else
  MEM_GB=$(( CPUS_PER_NODE * 2 ))
fi

# --exclusive: SLURM allocates (and CARC bills) every CPU, all memory and all GRES of each node. Cost
# the largest node the request could land on: 64 CPUs; 2 GPUs on A100/A40 nodes, 4 on L40S or when the
# type is open; 768 GB on L40S (or open-type GPU) nodes, 1.5 TB on largemem, 256 GB otherwise.
if [ "$EXCLUSIVE" -gt 0 ]; then
  _want=$(echo "$PART $GRES $GPN $GPT $GPUS $CONSTRAINT" | tr 'A-Z' 'a-z')
  [ "$CPUS_PER_NODE" -ge 64 ] || CPUS_PER_NODE=64
  case "$_want" in *largemem*) MEM_GB=1536;; *l40s*) MEM_GB=768;; *a100*|*a40*) MEM_GB=256;; *gpu*|*debug*) MEM_GB=768;; *) MEM_GB=256;; esac
  case "$_want" in *l40s*) _ng=4;; *a100*|*a40*) _ng=2;; *) _ng=4;; esac
  case "$PART" in *gpu*|*debug*) [ "$GPUS_PER_NODE" -ge "$_ng" ] || GPUS_PER_NODE=$_ng;; esac
fi

RATE_PER_NODE=$("$PY" -c "print(round($CPUS_PER_NODE + $MEM_GB * 0.25 + $GPUS_PER_NODE * $GW, 4))")   # SU per minute
PROJECTED=$("$PY" -c "print(round($RATE_PER_NODE * $NODES * $HOURS * 60 * $TASKS, 1))")

# ---------- 4. the checks ----------
USED=$(used_sus)
RESERVED=$(reserved_sus)
AFTER=$("$PY" -c "print(round($USED + $RESERVED + $PROJECTED, 1))")
echo "sbatch_guard: used=${USED} SU; reserved_by_active=${RESERVED} SU (${ACTIVE} jobs); projected_max=${PROJECTED} SU (${NODES} node x ${RATE_PER_NODE} SU/min x ${HOURS} h x ${TASKS} task(s); part=${PART} cpus/node=${CPUS_PER_NODE} mem/node=${MEM_GB}G gpus/node=${GPUS_PER_NODE} ${GPU_TYPE}$([ "$EXCLUSIVE" -gt 0 ] && echo ' exclusive')); after=${AFTER} / cap=${MAX_SUS}; active_jobs=${ACTIVE}/${MAX_JOBS}"

if [ -n "$PREFIX" ]; then
  case "$JOBNAME" in "$PREFIX"?*) ;;
    *) die "job name '${JOBNAME:-<unset: SLURM would use the script name>}' does not start with '${PREFIX}' (budget.yaml job_name_prefix): the SU ledger would not count it";;
  esac
fi
"$PY" -c "import sys; sys.exit(0 if $HOURS <= $MAX_HOURS else 1)" || die "time limit ${TIME} (${HOURS} h) exceeds max_job_hours=${MAX_HOURS}"
if [ "$GPUS_PER_NODE" -gt 0 ] && [ "$NODES" -gt "$MAX_GPU_NODES" ]; then die "${NODES} GPU nodes exceeds max_gpu_nodes_per_job=${MAX_GPU_NODES}"; fi
if [ $(( ACTIVE + TASKS )) -gt "$MAX_JOBS" ]; then die "${ACTIVE} jobs already running/pending + ${TASKS} new > max_concurrent_jobs=${MAX_JOBS} (every array task is a job)"; fi
"$PY" -c "import sys; sys.exit(0 if $AFTER <= $MAX_SUS else 1)" || die "used ${USED} + reserved ${RESERVED} + projected ${PROJECTED} = ${AFTER} SU would exceed max_project_sus=${MAX_SUS}"
"$PY" -c "import sys; sys.exit(0 if $AFTER <= $MAX_SUS*$WARN_FRAC else 1)" || echo "sbatch_guard: WARNING — past ${WARN_FRAC} of the SU cap" >&2
# sbatch runs in $PWD, which becomes the job's submit directory. For make submit that is a read-only code snapshot
# (scripts/snapshot.py): SLURM's default log (slurm-%j.out there) or a relative --output/--error could not be
# written, and the job would die without a log. A --chdir elsewhere would also hide the job from prune's in-use
# check (squeue %Z), so it must stay in the snapshot.
CHDIR=$(getopt_val chdir D)
if [ -f "$PWD/.lofi-snapshot" ] || [ ! -w "$PWD" ]; then
  case "$OUTPUT" in /*) ;; *) die "--output '${OUTPUT:-<unset>}' is not an absolute path: in its read-only submit directory ($PWD) the job could not write its log";; esac
  case "$ERROR" in ''|/*) ;; *) die "--error '${ERROR}' is not an absolute path: in its read-only submit directory ($PWD) the job could not write it";; esac
  if [ -n "$CHDIR" ] && [ "$(cd "$CHDIR" 2>/dev/null && pwd -P)" != "$(pwd -P)" ]; then
    die "--chdir '$CHDIR' moves the job out of its code snapshot $PWD"
  fi
fi
# SLURM does not create the --output/--error directory: a job whose log dir is missing dies without a trace
for _LOGF in "$OUTPUT" "$ERROR"; do
  [ -n "$_LOGF" ] || continue
  LOGDIR=$(dirname "$(printf '%s' "$_LOGF" | sed "s/%u/${USER}/g")")
  case "$LOGDIR" in
    /*%*|[!/]*) ;;   # a pattern left in the directory part, or a relative path: nothing to check
    *) [ -d "$LOGDIR" ] || mkdir -p "$LOGDIR" 2>/dev/null \
         || die "log directory ${LOGDIR} does not exist and cannot be created: the job would die without a log";;
  esac
done
# The ledger must be writable before anything is queued: a submission it misses is a job the SU log never saw
{ mkdir -p "$(dirname "$LOG")" && : >> "$LOG"; } 2>/dev/null || die "cannot write the ledger ${LOG}: nothing submitted"

# ---------- 4b. job-type registry and the soft SU target ----------
# A job type runs only if jobs/APPROVED.yaml lists it with the sha of its #SBATCH + non-comment lines
# (editing comments keeps the approval) and a max_su at least the projection. Entries above budget.yaml's
# self_approve_max_su carry "by: Shresth" (tests/test_approvals.py checks that).
if [ -n "$APPROVALS" ]; then
  case "$APPROVALS" in /*) APF="$APPROVALS";; *) APF="$GUARD_DIR/$APPROVALS";; esac
  SCRIPT_ABS="$(cd "$(dirname "$SCRIPT")" && pwd)/$(basename "$SCRIPT")"
  REG=$("$PY" - "$SCRIPT_ABS" "$GUARD_DIR" "$APF" "$PROJECTED" <<'PYREG'
import hashlib, os, re, sys
script, root, registry, projected = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rel = os.path.relpath(os.path.realpath(script), os.path.realpath(root))
keep = [l.rstrip() for l in open(script) if l.startswith("#SBATCH") or (l.strip() and not l.lstrip().startswith("#"))]
sha = hashlib.sha256("\n".join(keep).encode()).hexdigest()[:16]
entry = None
if os.path.exists(registry):
    for line in open(registry):
        m = re.match(r"^(\S+):\s*\{(.*)\}\s*$", line.strip())
        if m and m.group(1) == rel:
            entry = dict(kv.split(":", 1) for kv in m.group(2).split(",") if ":" in kv)
            entry = {k.strip(): v.strip() for k, v in entry.items()}
if entry is None:
    print(f"new {rel} sha={sha}")
elif entry.get("sha") != sha:
    print(f"changed {rel} sha={sha} approved_sha={entry.get('sha')}")
elif projected > float(entry.get("max_su", 0)):
    print(f"over-estimate {rel} projected={projected} max_su={entry.get('max_su')}")
else:
    print(f"ok {rel} sha={sha} max_su={entry.get('max_su')} by={entry.get('by')}")
PYREG
  ) || die "cannot check the job-type registry $APPROVALS"
  case "$REG" in
    ok\ *) echo "sbatch_guard: registry ${REG#ok }";;
    *) needs_approval "$REG projected=${PROJECTED} (then: make approve JOB=<job> MAX=<su> BY=Shresth)";;
  esac
fi
if [ -n "$TARGET" ] && [ -z "${PAST_TARGET_OK:-}" ]; then
  "$PY" -c "import sys; sys.exit(0 if $AFTER <= $TARGET else 1)" \
    || needs_approval "past-target used+reserved+projected=${AFTER} > target_sus=${TARGET} (set PAST_TARGET_OK=<date of Shresth's OK>)"
fi

# ---------- 5. submit ----------
OUT=$("$SBATCH_BIN" --account="$ACCOUNT" ${OPTS[@]+"${OPTS[@]}"} "$SCRIPT" ${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"})
echo "$OUT"
JOBID=$(echo "$OUT" | grep -oE '[0-9]+$' || true)
mkdir -p "$(dirname "$LOG")"
printf '%s\t%s\t%s\tused=%s\treserved=%s\tprojected=%s\tnodes=%s\tpart=%s\tgpus/node=%s\ttime=%s\ttasks=%s\n' \
  "$(date '+%F %T')" "${JOBID:-?}" "$SCRIPT" "$USED" "$RESERVED" "$PROJECTED" "$NODES" "$PART" "$GPUS_PER_NODE" "$TIME" "$TASKS" >> "$LOG"
