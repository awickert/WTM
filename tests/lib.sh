#!/usr/bin/env bash
# Shared helpers for the test suite. Source it:  . "$(dirname "$0")/../lib.sh"
#
# Growing file. Today it carries the run-log column lookup; the binary-identity guards
# (wtm_provenance / compare_binaries / bg_run) and make_work land here next.

# wtm_col <run-log> <column-name>  ->  the 1-based awk field number
#
# WHY NOT JUST WRITE $24. Because that is the failure this suite has been removing: the run log's
# columns are emitted once in src/WTM.cpp, and a literal field number in an awk one-liner is a silent
# reindex waiting for someone to insert a column. The Python side reads by name via wtm_log.py; this
# is the same guarantee for the shell side. Exits non-zero, naming the column, if it is not there --
# an awk field that does not exist is the empty string, which compares as 0 and passes quietly.
wtm_col() {
    local file="$1" want="$2" n
    n=$(awk -v want="$want" '/^Cycles_done/{for (i = 1; i <= NF; i++) if ($i == want) { print i; exit } }' "$file")
    if [ -z "$n" ]; then
        echo "wtm_col: run log '$file' has no column '$want'" >&2
        echo "  it has: $(awk '/^Cycles_done/{print; exit}' "$file")" >&2
        return 1
    fi
    printf '%s' "$n"
}
