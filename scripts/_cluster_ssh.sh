# _cluster_ssh.sh — sourced by remote.sh, pull_artifacts.sh and watch_job.sh (laptop side).
# Sets CLUSTER_HOST and CLUSTER_SSH_OPTS from kit.env.
#  * BatchMode: an expired Duo/ControlMaster session fails immediately instead of hanging the agent.
#  * Resolver fallback: on the USC VPN, macOS's system resolver sometimes stops resolving the cluster
#    name while plain DNS still answers. Then connect by address but keep host-key verification pinned
#    to the cluster's known key (HostKeyAlias), so this is never a security downgrade.
. "$(dirname "${BASH_SOURCE[0]}")/_kit_env.sh"
CLUSTER_HOST="$KIT_CLUSTER_HOST"
CLUSTER_SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=30"
if ! python3 -c 'import socket, sys; socket.getaddrinfo(sys.argv[1], 22)' "$KIT_CLUSTER_FQDN" >/dev/null 2>&1; then
  _ip=$(host "$KIT_CLUSTER_FQDN" 2>/dev/null | awk '/has address/ {print $4; exit}')
  if [ -n "$_ip" ]; then
    CLUSTER_SSH_OPTS="$CLUSTER_SSH_OPTS -o HostName=$_ip -o HostKeyAlias=$KIT_CLUSTER_FQDN"
    echo "[cluster-ssh] system resolver cannot resolve $KIT_CLUSTER_FQDN; using $_ip with the pinned host key" >&2
  fi
fi
