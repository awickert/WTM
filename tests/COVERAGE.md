# WTM test-coverage matrix

**Generated** by `tests/coverage_matrix.py` from the fingerprints WTM itself emits, so it reflects what each run RESOLVED to rather than what its config appears to say. Do not edit by hand; re-run the suite.

Runs recorded: **315** across **32** tests.


## 1. Combination coverage: every (solver, collector, integrator) against run type

The pairwise tables further down CANNOT answer this: two pairs can each be covered while their combination never runs. **0** means that combination has never been run at that run type.

| solver | collector | integrator | `equilibrium` | `test` | `transient` |
|---|---|---|---|---|---|
| `anderson` | `active_set` | `bdf2_on_V` | 10 | **0** | 5 |
| `anderson` | `active_set` | `be_volume` | 50 | 2 | 11 |
| `anderson` | `active_set` | `tr_bdf2` | 21 | **0** | 5 |
| `anderson` | `explicit` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `explicit` | `be_secant` | 9 | **0** | 1 |
| `anderson` | `explicit` | `be_volume` | 1 | **0** | 1 |
| `anderson` | `explicit` | `tr_bdf2` | 2 | **0** | 2 |
| `anderson` | `extended_soil` | `be_secant` | 2 | **0** | **0** |
| `anderson` | `implicit` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `implicit` | `be_secant` | 15 | **0** | 1 |
| `anderson` | `implicit` | `be_volume` | 3 | **0** | 1 |
| `anderson` | `implicit` | `tr_bdf2` | 3 | **0** | 2 |
| `anderson` | `legacy` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `legacy` | `be_secant` | 21 | **0** | 1 |
| `anderson` | `legacy` | `be_volume` | 1 | **0** | 1 |
| `anderson` | `legacy` | `tr_bdf2` | 2 | **0** | 2 |
| `anderson` | `off` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `off` | `be_secant` | 14 | **0** | 1 |
| `anderson` | `off` | `be_volume` | 1 | **0** | 1 |
| `anderson` | `off` | `tr_bdf2` | 2 | **0** | 2 |
| `newton` | `active_set` | `be_volume` | 8 | **0** | 4 |
| `newton` | `active_set` | `tr_bdf2` | 1 | **0** | 1 |
| `newton` | `explicit` | `be_secant` | 3 | **0** | 1 |
| `newton` | `explicit` | `be_volume` | 1 | **0** | 1 |
| `newton` | `explicit` | `tr_bdf2` | 1 | **0** | 1 |
| `newton` | `implicit` | `be_secant` | 3 | **0** | 1 |
| `newton` | `implicit` | `be_volume` | 1 | **0** | 1 |
| `newton` | `implicit` | `tr_bdf2` | 1 | **0** | 1 |
| `newton` | `legacy` | `be_secant` | 1 | **0** | 1 |
| `newton` | `legacy` | `be_volume` | 1 | **0** | 1 |
| `newton` | `legacy` | `tr_bdf2` | 1 | **0** | 1 |
| `newton` | `off` | `be_secant` | 1 | **0** | 1 |
| `newton` | `off` | `be_volume` | 1 | **0** | 1 |
| `newton` | `off` | `tr_bdf2` | 1 | **0** | 1 |
| `picard` | `explicit` | `bdf2_on_V` | 6 | **0** | 4 |
| `picard` | `explicit` | `be_secant` | 3 | **0** | 2 |
| `picard` | `explicit` | `be_volume` | 2 | **0** | 2 |
| `picard` | `implicit` | `bdf2_on_V` | 5 | **0** | 4 |
| `picard` | `implicit` | `be_secant` | 2 | **0** | 2 |
| `picard` | `implicit` | `be_volume` | 2 | **0** | 2 |
| `picard` | `legacy` | `bdf2_on_V` | 5 | **0** | 4 |
| `picard` | `legacy` | `be_secant` | 2 | **0** | 2 |
| `picard` | `legacy` | `be_volume` | 2 | **0** | 2 |
| `picard` | `off` | `bdf2_on_V` | 4 | **0** | 4 |
| `picard` | `off` | `be_secant` | 2 | **0** | 2 |
| `picard` | `off` | `be_volume` | 2 | **0** | 2 |

**46** distinct combinations are exercised at all. Of those, **45** run in BOTH equilibrium and transient, **1** are equilibrium-only and **0** transient-only.

Equilibrium-only, i.e. never exercised on the transient path:

- `anderson` x `extended_soil` x `be_secant`


## 2. What each test covers

| test | run_type | solver | integrator | dtctl | collector | fsm | runoff_ratio | ranks |
|---|---|---|---|---|---|---|---|---|
| `FSM_MPI_consistency` | equilibrium | anderson | be_volume | fixed | active_set | 1 | 0 | 1,4 |
| `FSM_conservation_+_lake` | equilibrium | anderson | be_volume | fixed | active_set | 1 | 0 | 1 |
| `MPI_consistency_matrix` | equilibrium | anderson | be_volume | fixed | active_set | 0,1 | 0,1 | 1,4 |
| `Newton_Jacobian_+_contract` | equilibrium | anderson,newton | be_secant,be_volume | continuation,fixed | active_set,explicit,implicit | 1 | 0 | 1 |
| `active-set_collector-indep` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,explicit,implicit | 1 | 0 | 1 |
| `adaptive-restart_robustness` | equilibrium | anderson | be_volume | fixed | active_set | 0 | 0 | 1 |
| `adaptive_dt_+_water_metric` | equilibrium | anderson | be_volume,tr_bdf2 | adaptive,fixed | active_set | 0 | 0 | 1 |
| `adaptive_estimator_order` | equilibrium | anderson | bdf2_on_V,tr_bdf2 | adaptive | active_set | 0,1 | 1 | 1 |
| `boundary:_analytic_parabola` | equilibrium | anderson | be_secant | fixed | off | 0 | 0 | 1 |
| `boundary:_dirichlet≡padding` | equilibrium | anderson,newton | be_secant | fixed | explicit | 0 | 0 | 1 |
| `cascade_A->B->ocean_(skim)` | equilibrium | anderson | be_volume | fixed | active_set | 1 | 0 | 1,4 |
| `combination_sweep` | equilibrium,transient | anderson,newton,picard | bdf2_on_V,be_secant,be_volume,tr_bdf2 | continuation,fixed | active_set,explicit,implicit,legacy,off | 1 | 0 | 1 |
| `dt-sensitivity_(active-set)` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,legacy | 0 | 0 | 1 |
| `flicker_1:_storativity_jump` | transient | anderson | bdf2_on_V,be_volume | fixed | active_set | 0 | 0 | 1 |
| `flicker_2:_evap_discontinuity` | equilibrium | anderson | be_secant | fixed | implicit | 0 | 0 | 1 |
| `ghost-boundary_(#96)` | transient | anderson,newton | bdf2_on_V,be_volume,tr_bdf2 | fixed | active_set | 0 | 0 | 1,4 |
| `ghost-cell_MPI` | equilibrium | anderson | be_volume | fixed | active_set | 0 | 0 | 1,2 |
| `golden_(expected_results)` | equilibrium,transient | anderson | be_volume | fixed | active_set | 0,1 | 0,1 | 1,4 |
| `local-in-space_water_ledger` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,off | 0 | 0,1 | 1 |
| `mass-balance_MPI` | test | anderson | be_volume | fixed | active_set | 1 | 0 | 1,4 |
| `multi-lake_stages_vs_dt` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,implicit | 1 | 0 | 4 |
| `nested_DH_+_skim_spill-accuracy` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,implicit | 1 | 0 | 1,4 |
| `recharge_consistency_(#93)` | transient | anderson | bdf2_on_V,be_volume,tr_bdf2 | fixed | active_set | 0 | 0 | 1 |
| `runoff_collector_selector` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,explicit,extended_soil,implicit,off | 0 | 0 | 1 |
| `runoff_gathering_(wtd=0)` | equilibrium | anderson | be_secant | fixed | implicit,off | 0 | 0 | 1 |
| `serial_rank-0_recharge_path` | equilibrium | anderson | be_volume | fixed | active_set | 1 | 1 | 1,4 |
| `snapshot_name_+_restart` | equilibrium | anderson | be_secant | fixed | implicit | 0 | 0 | 1 |
| `solve-count_invariance` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set | 1 | 0,1 | 1 |
| `solver_consistency_(A≡P≡N)` | equilibrium | anderson,newton,picard | be_secant,be_volume | continuation,fixed | active_set,explicit | 0 | 0 | 1 |
| `storage_secant≡volume` | transient | anderson | be_volume | fixed | active_set | 0 | 0 | 1 |
| `taper_determinism+smooth` | equilibrium | anderson | be_secant | fixed | legacy | 0,1 | 0 | 1,4 |
| `water-budget_closure_(schemes)` | equilibrium | anderson,newton,picard | bdf2_on_V,be_secant,be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit,legacy,off | 1 | 1 | 1 |

## 3. Pairwise crossings

A blank cell is a combination **no run exercises**. `by design` and `does not converge` are LEARNED from tests/combination_sweep, which attempts every combination and records what it did -- so those are observations, not assertions.


### solver x collector

| solver \ collector | active_set | explicit | extended_soil | implicit | legacy | off |
|---|---|---|---|---|---|---|
| **anderson** | 104 | 18 | 2 | 27 | 30 | 23 |
| **newton** | 14 | 8 |   | 8 | 6 | 6 |
| **picard** | by design | 19 |   | 17 | 17 | 16 |

### integrator x collector

| integrator \ collector | active_set | explicit | extended_soil | implicit | legacy | off |
|---|---|---|---|---|---|---|
| **bdf2_on_V** | 15 | 12 |   | 11 | 11 | 10 |
| **be_secant** | by design | 19 | 2 | 24 | 28 | 21 |
| **be_volume** | 75 | 8 |   | 10 | 8 | 8 |
| **tr_bdf2** | 28 | 6 |   | 7 | 6 | 6 |

### dtctl x collector

| dtctl \ collector | active_set | explicit | extended_soil | implicit | legacy | off |
|---|---|---|---|---|---|---|
| **adaptive** | 25 |   |   |   |   |   |
| **continuation** | 10 | 6 |   | 7 | 6 | 6 |
| **fixed** | 83 | 39 | 2 | 45 | 47 | 39 |

### run_type x collector

| run_type \ collector | active_set | explicit | extended_soil | implicit | legacy | off |
|---|---|---|---|---|---|---|
| **equilibrium** | 90 | 29 | 2 | 36 | 37 | 29 |
| **test** | 2 |   |   |   |   |   |
| **transient** | 26 | 16 |   | 16 | 16 | 16 |

### solver x integrator

| solver \ integrator | bdf2_on_V | be_secant | be_volume | tr_bdf2 |
|---|---|---|---|---|
| **anderson** | 23 | 65 | 73 | 43 |
| **newton** |   | 12 | 20 | 10 |
| **picard** | 36 | 17 | 16 |   |

### run_type x solver

| run_type \ solver | anderson | newton | picard |
|---|---|---|---|
| **equilibrium** | 161 | 25 | 37 |
| **test** | 2 |   |   |
| **transient** | 41 | 17 | 32 |

### fsm x collector

| fsm \ collector | active_set | explicit | extended_soil | implicit | legacy | off |
|---|---|---|---|---|---|---|
| **0** | 42 | 8 | 2 | 7 | 5 | 12 |
| **1** | 76 | 37 |   | 45 | 48 | 33 |

### runoff_ratio x dtctl

| runoff_ratio \ dtctl | adaptive | continuation | fixed |
|---|---|---|---|
| **0** | 3 | 33 | 225 |
| **1** | 22 | 2 | 30 |

## 4. Uncovered pairwise crossings

**22** combinations are reachable but exercised by nothing:

- `solver=newton` x `collector=extended_soil`
- `solver=picard` x `collector=extended_soil`
- `integrator=bdf2_on_V` x `collector=extended_soil`
- `integrator=be_volume` x `collector=extended_soil`
- `integrator=tr_bdf2` x `collector=extended_soil`
- `dtctl=adaptive` x `collector=explicit`
- `dtctl=adaptive` x `collector=extended_soil`
- `dtctl=adaptive` x `collector=implicit`
- `dtctl=adaptive` x `collector=legacy`
- `dtctl=adaptive` x `collector=off`
- `dtctl=continuation` x `collector=extended_soil`
- `run_type=test` x `collector=explicit`
- `run_type=test` x `collector=extended_soil`
- `run_type=test` x `collector=implicit`
- `run_type=test` x `collector=legacy`
- `run_type=test` x `collector=off`
- `run_type=transient` x `collector=extended_soil`
- `solver=newton` x `integrator=bdf2_on_V`
- `solver=picard` x `integrator=tr_bdf2`
- `run_type=test` x `solver=newton`
- `run_type=test` x `solver=picard`
- `fsm=1` x `collector=extended_soil`

Each is a place a defect could live unseen. That is not a demand to cover all of them -- some are uninteresting -- but the list should be read, and anything load-bearing should get an arm.

