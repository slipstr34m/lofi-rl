#!/usr/bin/env bash
# ssh_cluster.sh: ssh with the cluster options from _cluster_ssh.sh (BatchMode, resolver fallback).
# The Makefile's rsync calls use it via `rsync -e scripts/ssh_cluster.sh`.
. "$(dirname "$0")/_cluster_ssh.sh"
exec ssh $CLUSTER_SSH_OPTS "$@"
