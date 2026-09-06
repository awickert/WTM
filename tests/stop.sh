#!/usr/bin/env bash
# Stop a live tests/run_all.sh BY PID, read from .run_all.lock.
#
# WHY THIS EXISTS: `pkill -f <pattern>` twice matched the caller's OWN command line and killed the
# calling shell (exit 144). A pattern kill has no way to know which processes it is entitled to.
set -uo pipefail
cd "$(dirname "$0")"
LOCK=.run_all.lock
[ -f "$LOCK" ] || { echo "no suite run is recorded in $LOCK"; exit 0; }
PID=$(head -1 "$LOCK"); COMMIT=$(sed -n 2p "$LOCK")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "stale lock (PID $PID, commit $COMMIT, not running); removing"; rm -f "$LOCK"; exit 0
fi
echo "stopping suite PID $PID (commit $COMMIT) and its process group"
kill -TERM -"$(ps -o pgid= -p "$PID" | tr -d ' ')" 2>/dev/null || kill -TERM "$PID"
sleep 2
kill -0 "$PID" 2>/dev/null && { echo "still alive; SIGKILL"; kill -KILL -"$(ps -o pgid= -p "$PID" | tr -d ' ')" 2>/dev/null; }
rm -f "$LOCK"; echo "stopped"
