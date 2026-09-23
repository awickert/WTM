# Which coupling scheme does each suite test, and why

`surface_water.coupling.iterations` has two regimes -- the LAGGED scheme (`1`, a step uses the
PREVIOUS step's FillSpillMerge output) and the ITERATED default (a step re-solves against its own,
stopping when the state stops moving). Both are supported, so every suite that can tell them apart
has to say which one it means, and for what reason.

Before this file the answer was an accumulation: a dedicated suite, one suite with dual arms, and a
blanket re-run of seven others driven by an environment variable. Three mechanisms and no rule.

## THE RULE

**A suite's coupling coverage is decided by what its assertions CLAIM -- never by which hook fits.**

| category | the claim | where the scheme comes from |
|---|---|---|
| **MECHANISM** | the feature itself works | per-arm, the scheme IS the subject |
| **COMPARATIVE** | a claim that spans BOTH schemes | in-suite dual arms; one invocation cannot compare |
| **REPRODUCIBILITY** | "the answer has not changed" | pinned explicitly, per arm, and kept pinned |
| **INVARIANT** | holds under either scheme | THE SHIPPED DEFAULT -- a suite certifies what ships |
| **INERT** | `routing: impulse` or `off` | the key cannot matter; it resolves to 1 |

The INVARIANT row is the one that was wrong before 2026-09-23: 40 of 41 configs pinned `1`, which
had stopped being the default, so the suite predominantly certified a model users do not get.

## THE CLASSIFICATION

Eighteen suites can tell the schemes apart -- ten state `routing: continuous` outright and eight
template it with a `continuous` arm. The rest are INERT and need no thought beyond declaring the key.

| suite | category | why |
|---|---|---|
| `coupling_iteration` | MECHANISM | the scheme is its subject; arms at 1, 2 and 3 passes |
| `multilake` | COMPARATIVE | asserts "iterating is never worse" and "the schemes converge on each other" -- both span the two, so it runs arms `A*` lagged and `B*` iterated itself |
| `golden` | REPRODUCIBILITY | its lagged arms pin every pre-#112 result and STAY; new arms certify the iterated default beside them (Andy, 2026-09-23) |
| `active_set` | INVARIANT | lake persistence, collector distinctness -- properties, not scheme artefacts |
| `fsm_cascade` | INVARIANT | chain sill levels, conservation, MPI identity |
| `fsm_consistency` | INVARIANT | cross-decomposition agreement |
| `fsm_fullness` | INVARIANT | depression hierarchy, spill level, skim vs plain |
| `lake_evap_equals_et` | INVARIANT | the ET/open-water transition vanishes when they are equal |
| `taper` | INVARIANT | taper determinism and smoothness; nothing about the coupling |
| `xrank_adaptive` | INVARIANT | cross-rank determinism of the adaptive controller |
| `xrank_growth` | INVARIANT | cross-rank drift stays bounded |
| `budget_step_ledger` | INVARIANT | per-step water ledger closes |
| `estimator_order` | INVARIANT | the adaptive estimator's observed order in dt |
| `fsm_exit_path` | INVARIANT | coarse run vs its own 1/1000 yr reference |
| `mpi_consistency` | INVARIANT | rank-count independence |
| `budget_closure` | INVARIANT | closure across schemes and collectors; its subject is ROUTING, not the pass count |
| `coupling_convergence` | INVARIANT | how FSM's result reaches the solver -- again routing, not the pass count |
| `newton_solver` | INVARIANT | Jacobian and contract; routing is one dial among several |

## WHAT THIS REPLACED

The `WTM_TEST_ITERATIONS` override and its `apply_test_iterations` helper existed to run seven
suites a second time under the default. It did its job -- it is what found multilake's band -- but as
a permanent structure it was the wrong shape: the knowledge that a suite ran twice lived only in
`run_all.sh`, a suite with mixed routing arms could be made vacuous by it (`active_set` and
`xrank_growth` both were, on the first attempt), and a passing re-run mostly proved the suite still
worked rather than that the scheme was right. With INVARIANT suites declaring the default outright,
the override has nothing left to do.
