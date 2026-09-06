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

# ---------------------------------------------------------------------------------------------------
# BINARY IDENTITY. A measurement is only as trustworthy as the knowledge of WHICH BINARY produced it.
# This is not abstract: a 1.43 m golden discrepancy was attributed to a convergence-metric change when
# the comparison had actually been run against a binary still carrying an unrelated controller defect,
# and the wrong conclusion reached a commit message, the CHANGELOG and a memory note before it was
# caught. These helpers make that class of mistake refuse to happen rather than rely on remembering.

# wtm_binary_commit <binary>  ->  "<commit> <clean|dirty>"
wtm_binary_commit() { "$1" -wtm_version 2>/dev/null | awk '/^wtm_git_commit/{c=$2} /^wtm_git_state/{s=$2} END{print c, s}'; }

# wtm_stale <binary>  ->  0 (true) if any tracked source is NEWER than the binary
#
# STALENESS IS CHECKED BY MTIME, NOT BY COMPARING THE STAMP TO HEAD. The stamp records the commit at
# BUILD time, so a tests-only or docs-only commit moves HEAD without invalidating the binary at all --
# a commit comparison would cry wolf on nearly every run and be switched off within a day. Observed
# live: a binary stamped bce7cc8 while HEAD was several tests-only commits ahead, and it was current.
wtm_stale() {
    local bin="$1" root newer
    # Resolve the repo root from THIS FILE, not from $0. When lib.sh is SOURCED, $0 is the calling
    # shell ("bash"), so dirname "$0" is "." and the path landed outside the repo -- find then had
    # nothing to look at and the guard silently reported "not stale" for everything. Caught by
    # probing it rather than by reading it. BASH_SOURCE is correct whether sourced or executed.
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd) || return 1
    [ -x "$bin" ] || return 1
    newer=$(find "$root/src" -newer "$bin" -name '*.cpp' -o -newer "$bin" -name '*.hpp' 2>/dev/null | head -3)
    [ -n "$newer" ] && { printf '%s\n' "$newer"; return 0; }
    return 1
}

# wtm_provenance <binary>   Announce which build this is; FAIL only on staleness.
#
# THE DIRTY/STALE DISTINCTION IS DELIBERATE, and it is a considered departure from "fail on dirty".
# STALE is unambiguously an error: source is newer than the binary, so you are testing code you did
# not build, and the run means nothing. DIRTY is merely unreproducible-from-the-hash -- and it is the
# NORMAL state while iterating, which is exactly when the suite is run most. Failing on it would
# train everyone to export WTM_ALLOW_DIRTY=1 permanently and the guard would be gone. So dirty is
# recorded loudly and does not block; compare_binaries below IS strict about it, because an A/B is
# not the routine case and is precisely where an uncommitted difference misleads.
wtm_provenance() {
    local bin="${1:-../build/wtm.x}" info commit state newer
    info=$(wtm_binary_commit "$bin"); commit=${info% *}; state=${info#* }
    echo "  binary: $bin"
    echo "  built from: ${commit:0:12} ($state)"
    [ "$state" = dirty ] && echo "  NOTE: built from a DIRTY tree -- these results are not reproducible from the hash alone."
    if newer=$(wtm_stale "$bin"); then
        echo "  ERROR: source is NEWER than the binary -- you would be testing code you did not build:" >&2
        printf '    %s\n' $newer >&2
        echo "    rebuild first (cmake --build build)" >&2
        return 1
    fi
    return 0
}

# compare_binaries <A> <B>   Refuse an A/B that cannot mean what it claims.
compare_binaries() {
    local a="$1" b="$2" ia ib rc=0
    ia=$(wtm_binary_commit "$a"); ib=$(wtm_binary_commit "$b")
    echo "  A: $a  ${ia}"
    echo "  B: $b  ${ib}"
    if [ "${ia% *}" = "${ib% *}" ]; then
        echo "  REFUSING: both binaries were built from the SAME commit (${ia% *}); this A/B compares a" >&2
        echo "  build with itself and any difference you find is noise." >&2
        rc=1
    fi
    for x in "$a" "$b"; do
        [ "$(wtm_binary_commit "$x" | awk '{print $2}')" = dirty ] && {
            echo "  REFUSING: $x was built from a DIRTY tree, so what it contains is not knowable from" >&2
            echo "  its hash -- commit or stash first. (WTM_ALLOW_DIRTY=1 to override deliberately.)" >&2
            [ "${WTM_ALLOW_DIRTY:-0}" = 1 ] || rc=1; }
        wtm_stale "$x" >/dev/null && { echo "  REFUSING: $x is stale against src/" >&2; rc=1; }
    done
    return $rc
}

# bg_run <dir> <cmd...>   Background a job with its directory captured AT CALL TIME.
#
# A backgrounded job inherits whatever cwd it happens to start in. Launching one without an explicit
# cd is how a run intended for an isolated worktree executed in the main tree and began writing its
# output into a file labelled "baseline".
bg_run() {
    local dir="$1"; shift
    [ -d "$dir" ] || { echo "bg_run: no such directory: $dir" >&2; return 1; }
    ( cd "$dir" && "$@" ) &
}

# expect_resolved <coverage-log> key=value [key=value ...]
#
# Assert that the MOST RECENT run resolved to what the arm asked for. The model itself writes that
# line (src/transient_groundwater.cpp::emit_coverage_fingerprint) AFTER every override, downgrade and
# auto-enable, so it records what the run ACTUALLY did rather than what a config appears to say.
#
# WHY THIS MATTERS MORE THAN IT SOUNDS: the fingerprint exists because "twice during this work a sed
# meant to switch a collector silently did nothing and a whole measurement was made on the wrong
# configuration". Until now it only fed COVERAGE.md, which run_all.sh calls "a map, not a gate". This
# turns it into a gate for the arms that care. It is also the antidote to the vacuous-arm class: an
# arm that quietly inherits a default -- because a key was misspelled, or a default later moved -- is
# testing the control twice and proving nothing, and only the model can say so.
#
# Keyed on the LAST coverage line rather than on the test tag: tags contain spaces, so `test=<tag>`
# cannot be tokenised unambiguously, whereas "the run I just did" is exact. Call it straight after
# the run whose resolution you mean.
expect_resolved() {
    local log="$1"; shift
    local line rc=0 pair k v got
    [ -s "$log" ] || { echo "  FAIL  RESOLVED  no coverage fingerprints in $log (is WTM_COVERAGE_LOG set?)" >&2; return 1; }
    line=$(grep '^coverage ' "$log" | tail -1)
    for pair in "$@"; do
        k=${pair%%=*}; v=${pair#*=}
        # tokenise on whitespace, then split at the FIRST '=' -- substring-proof, as in wtm_log.py
        got=$(printf '%s\n' $line | awk -F= -v k="$k" '$1==k {print substr($0, index($0,"=")+1); exit}')
        if [ "$got" != "$v" ]; then
            echo "  FAIL  RESOLVED  the run resolved $k=${got:-<absent>}, but the arm asked for $k=$v" >&2
            echo "                  full fingerprint: $line" >&2
            rc=1
        fi
    done
    [ $rc -eq 0 ] && echo "  OK   RESOLVED  the run actually used: $*"
    return $rc
}
