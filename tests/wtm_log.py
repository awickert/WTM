"""Read WTM's run log and its trace lines BY NAME, never by position.

TWO FAILURES THIS EXISTS TO REMOVE.

1. POSITIONAL COLUMN INDICES. The run log's header is emitted once, in src/WTM.cpp, and tests were
   reading it as `r[8]`, `r[16]` and so on. Nothing asserted the header-to-index mapping, so
   inserting a column would silently reindex every budget assertion in the suite -- and several
   would still report PASS, because a residual column read as a recharge column is still a number.
   The failure would not announce itself; it would just quietly change what the suite believes.
   read_log() returns named columns and raises on a name it cannot find.

2. SUBSTRING COLLISIONS IN TRACE PARSING. DTTRACE emits `est=`, `eint=`, `ecpl=`, `nest=`, `ncpl=`.
   A regex for `est=` also matches inside `nest=`, which is not hypothetical: it produced a phantom
   "step-count doubling" in a real investigation, and tests/estimator_order escapes it only by
   accident of adjacency. read_trace() tokenizes on WHITESPACE first and only then splits each token
   at its FIRST '=', so a key can never match inside another key. That is a property of the parse,
   not a property of the pattern, which is why it cannot regress.
"""


def _data_rows(path):
    """The numeric data rows. The log also carries a header and trailing non-numeric lines."""
    for line in open(path, errors="ignore"):
        line = line.strip()
        if line and line[0].isdigit():
            yield line.split()


def header_names(path):
    """The column names, in order, as the run actually wrote them."""
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("Cycles_done"):
                return line.split()
    raise AssertionError("no run-log header (a line starting 'Cycles_done') in %s" % path)


class RunLog:
    def __init__(self, path):
        self.path = path
        self.names = header_names(path)
        self.rows = [r for r in _data_rows(path)]

    def col(self, name):
        """One column as floats. Raises -- loudly, by name -- if the log has no such column."""
        if name not in self.names:
            raise KeyError("run log %s has no column %r.\n  it has: %s"
                           % (self.path, name, " ".join(self.names)))
        i = self.names.index(name)
        out = []
        for r in self.rows:
            if i >= len(r):
                raise AssertionError("run log %s: row has %d fields, column %r is at index %d\n  row: %s"
                                     % (self.path, len(r), name, i, " ".join(r)))
            out.append(float(r[i]))
        return out

    def last(self, name):
        c = self.col(name)
        if not c:
            raise AssertionError("run log %s has no data rows" % self.path)
        return c[-1]


def read_log(path):
    return RunLog(path)


def read_trace(path, tag):
    """Every `<tag> k=v k=v ...` line, as a list of dicts of strings.

    Substring-proof by construction: split on whitespace, then at the FIRST '=' of each token.
    """
    out = []
    for line in open(path, errors="ignore"):
        if not line.startswith(tag + " "):
            continue
        d = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        out.append(d)
    return out


def trace_floats(path, tag, *keys):
    """read_trace, projected onto the named keys as floats; raises on a key that is never present."""
    rows = read_trace(path, tag)
    if not rows:
        raise AssertionError("no %s lines in %s" % (tag, path))
    for k in keys:
        if k not in rows[0]:
            raise KeyError("%s lines in %s have no key %r.\n  they have: %s"
                           % (tag, path, k, " ".join(sorted(rows[0]))))
    return [tuple(float(r[k]) for k in keys) for r in rows]
