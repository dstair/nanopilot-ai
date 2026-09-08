#!/bin/bash
set -uo pipefail

pkill -f drone_manage 2>/dev/null || true
pkill -9 -f betaflight_SITL 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "ruby.*gz" 2>/dev/null || true
pkill -9 -f gz_camera_worker 2>/dev/null || true
pkill -f "XboxBridge.app|Contents/MacOS/XboxBridge" 2>/dev/null || true
rm -f /tmp/nanopilot_xbox.sock 2>/dev/null

# The kills above only reach start.sh's CHILDREN. start.sh itself parks on a
# `tail -f /dev/null` keepalive, so every `--no-manage` launch used to leave a
# shell alive forever (18 had piled up by 2026-08-24, which also breaks the
# "is a run in progress?" check in the fresh-session checklist).
#
# Never kill a start.sh that is calling us: it runs this script from its own
# EXIT trap, so its pid is an ancestor of this shell. Kill the keepalive child
# first, then the wrapper, or the child is orphaned and survives anyway.
ancestors=" "
p=$$
while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    ancestors="$ancestors$p "
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
done

# ANCHORED: match only a shell actually EXECUTING start.sh, i.e. the command
# line begins with a bash invoking it. An unanchored "scripts/start.sh" also
# matches any process that merely MENTIONS the path -- a pgrep monitoring the
# stack, an editor, a wrapper script -- and killing those is how this script
# once killed a training run that grepped for stack processes.
for pid in $(pgrep -f '^[^ ]*/?bash[^ ]* [^ ]*scripts/start\.sh' 2>/dev/null || true); do
    case "$ancestors" in *" $pid "*) continue ;; esac
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
done
