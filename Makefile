# lofi-rl Makefile. Laptop only: the cluster never runs make (CLAUDE.md §1, §3).
# GNU rsync 3.x is required (macOS /usr/bin/rsync is openrsync, protocol 29): brew install rsync.
SHELL := /bin/bash
include kit.env

RSYNC := /opt/homebrew/bin/rsync
SSH := $(CURDIR)/scripts/ssh_cluster.sh
RS := $(RSYNC) -az -e $(SSH)
SYNC := $(RS) --delete --max-delete=200 --exclude-from=.rsyncignore
PY := .venv/bin/python
HOST := $(KIT_CLUSTER_HOST)

.PHONY: help sync pull test status submit facts init approve backup-votes push-assets

help:
	@awk -F':.*## ' '/^[a-z-]+:.*## /{printf "  %-13s %s\n", $$1, $$2}' Makefile

sync: ## push the code to the cluster mirror (rsync --delete; refuses to delete protected paths)
	@$(RSYNC) --version | head -1 | grep -q 'version 3\.' || { echo "GNU rsync 3.x required: brew install rsync"; exit 1; }
	@scripts/check_frozen.sh
	@scripts/stamp.sh > .sync-stamp
	@set -o pipefail; $(SYNC) -n -i ./ $(HOST):$(REMOTE_PROJ)/ | scripts/check_deletes.sh
	@$(SYNC) ./ $(HOST):$(REMOTE_PROJ)/
	@echo "synced $$(cut -d' ' -f1-3 .sync-stamp) -> $(REMOTE_PROJ)"

pull: ## pull run summaries (runs/<run>/summary/, structure kept) and inbox clips; never deletes locally
	@mkdir -p runs inbox
	$(RS) -m --include='/*/' --include='/*/summary/***' --exclude='*' $(HOST):$(REMOTE_SCR)/runs/ runs/
	$(RS) $(HOST):$(REMOTE_SCR)/inbox/ inbox/

test: ## the commit oracle (CLAUDE.md §6)
	@set -o pipefail; $(PY) -m pytest -q

status: ## queue, last guard submissions, SU ledger total
	@scripts/remote.sh 'squeue -u $$USER -o "%.10i %.24j %.8T %.9M %.9l %R"; tail -n 3 $(REMOTE_KEEP)/submissions.log 2>/dev/null || true; python3 scripts/session_report.py | tail -n 1'

submit: ## make submit JOB=jobs/x.sbatch [OVR='--dependency=afterok:123'] [ARGS='...']
	@[ -n "$(JOB)" ] || { echo "usage: make submit JOB=jobs/x.sbatch [OVR=...] [ARGS=...]"; exit 2; }
	scripts/submit.sh $(OVR) $(JOB) $(ARGS)

facts: ## read-only cluster facts (paste into the CHANGELOG)
	scripts/cluster_setup.sh facts

init: ## create the cluster directories (idempotent)
	scripts/cluster_setup.sh init

approve: ## make approve JOB=jobs/x.sbatch MAX=<su> BY=<Shresth|Claude>
	$(PY) scripts/approve_job.py $(JOB) --max-su $(MAX) --by $(BY)

backup-votes: ## commit+push data/votes (private repo), then copy to the cluster KEEP dir (never --delete)
	@[ -d data/votes/.git ] || { echo "data/votes is not the private votes repo yet (see docs/DECISIONS.md D-010)"; exit 1; }
	cd data/votes && git add -A && { git diff --cached --quiet || git commit -qm "votes $$(date +%F_%H%M)"; } && git push -q
	$(RS) data/votes/ $(HOST):$(REMOTE_KEEP)/votes/

.PHONY: rate
rate: ## serve the rating UI on 127.0.0.1:8596 (BATCH=inbox/<date>/<batch>)
	@PYTHONPATH=src $(PY) -m lofi_rl.ui.serve $(if $(BATCH),"$(BATCH)",)

push-assets: ## copy assets/big/ (fetch-only sources) to $REMOTE_KEEP/assets/: no --delete, no .rsyncignore, owner-only
	$(RS) --chmod=Dgo-rwx,Fgo-rwx --exclude=.DS_Store assets/big/ $(HOST):$(REMOTE_KEEP)/assets/
