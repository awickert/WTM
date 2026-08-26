# WTM test-coverage matrix

**Generated** by `tests/coverage_matrix.py` from the fingerprints WTM itself emits, so it reflects what each run RESOLVED to rather than what its config appears to say. Do not edit by hand; re-run the suite.

Runs recorded: **144** across **30** tests.


## 1. Combination coverage: every (solver, collector, integrator) against run type

The pairwise tables further down CANNOT answer this: two pairs can each be covered while their combination never runs. **0** means that combination has never been run at that run type.

| solver | collector | integrator | `equilibrium` | `test` | `transient` |
|---|---|---|---|---|---|
| `anderson` | `active_set` | `bdf2_on_V` | 1 | **0** | 4 |
| `anderson` | `active_set` | `be_volume` | 48 | 2 | 9 |
| `anderson` | `active_set` | `tr_bdf2` | 9 | **0** | 3 |
| `anderson` | `explicit` | `be_secant` | 7 | **0** | **0** |
| `anderson` | `implicit` | `be_secant` | 14 | **0** | **0** |
| `anderson` | `implicit` | `be_volume` | 2 | **0** | **0** |
| `anderson` | `implicit` | `tr_bdf2` | 1 | **0** | **0** |
| `anderson` | `legacy` | `be_secant` | 20 | **0** | **0** |
| `anderson` | `off` | `be_secant` | 7 | **0** | **0** |
| `newton` | `active_set` | `be_volume` | 6 | **0** | 2 |
| `newton` | `explicit` | `be_secant` | 2 | **0** | **0** |
| `newton` | `implicit` | `be_secant` | 2 | **0** | **0** |
| `picard` | `explicit` | `bdf2_on_V` | 2 | **0** | **0** |
| `picard` | `explicit` | `be_secant` | 1 | **0** | **0** |
| `picard` | `implicit` | `bdf2_on_V` | 1 | **0** | **0** |
| `picard` | `legacy` | `bdf2_on_V` | 1 | **0** | **0** |

**16** distinct combinations are exercised at all. Of those, **4** run in BOTH equilibrium and transient, **12** are equilibrium-only and **0** transient-only.

Equilibrium-only, i.e. never exercised on the transient path:

- `anderson` x `explicit` x `be_secant`
- `anderson` x `implicit` x `be_secant`
- `anderson` x `implicit` x `be_volume`
- `anderson` x `implicit` x `tr_bdf2`
- `anderson` x `legacy` x `be_secant`
- `anderson` x `off` x `be_secant`
- `newton` x `explicit` x `be_secant`
- `newton` x `implicit` x `be_secant`
- `picard` x `explicit` x `bdf2_on_V`
- `picard` x `explicit` x `be_secant`
- `picard` x `implicit` x `bdf2_on_V`
- `picard` x `legacy` x `bdf2_on_V`


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
| `boundary:_analytic_parabola` | equilibrium | anderson | be_secant | fixed | off | 0 | 0 | 1 |
| `boundary:_dirichlet≡padding` | equilibrium | anderson,newton | be_secant | fixed | explicit | 0 | 0 | 1 |
| `cascade_A->B->ocean_(skim)` | equilibrium | anderson | be_volume | fixed | active_set | 1 | 0 | 1,4 |
| `dt-sensitivity_(active-set)` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,legacy | 0 | 0 | 1 |
| `flicker_1:_storativity_jump` | transient | anderson | bdf2_on_V,be_volume | fixed | active_set | 0 | 0 | 1 |
| `flicker_2:_evap_discontinuity` | equilibrium | anderson | be_secant | fixed | implicit | 0 | 0 | 1 |
| `ghost-boundary_(#96)` | transient | anderson,newton | bdf2_on_V,be_volume,tr_bdf2 | fixed | active_set | 0 | 0 | 1,4 |
| `ghost-cell_MPI` | equilibrium | anderson | be_volume | fixed | active_set | 0 | 0 | 1,2 |
| `golden_(expected_results)` | equilibrium,transient | anderson | be_volume | fixed | active_set | 0,1 | 0,1 | 1,4 |
| `local-in-space_water_ledger` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,off | 0 | 0 | 1 |
| `mass-balance_MPI` | test | anderson | be_volume | fixed | active_set | 1 | 0 | 1,4 |
| `multi-lake_stages_vs_dt` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,implicit | 1 | 0 | 4 |
| `nested_DH_+_skim_spill-accuracy` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,implicit | 1 | 0 | 1,4 |
| `recharge_consistency_(#93)` | transient | anderson | bdf2_on_V,be_volume,tr_bdf2 | fixed | active_set | 0 | 0 | 1 |
| `runoff_collector_selector` | equilibrium | anderson | be_secant,be_volume | fixed | active_set,explicit,implicit,off | 0 | 0 | 1 |
| `runoff_gathering_(wtd=0)` | equilibrium | anderson | be_secant | fixed | implicit,off | 0 | 0 | 1 |
| `serial_rank-0_recharge_path` | equilibrium | anderson | be_volume | fixed | active_set | 1 | 1 | 1,4 |
| `snapshot_name_+_restart` | equilibrium | anderson | be_secant | fixed | implicit | 0 | 0 | 1 |
| `solve-count_invariance` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set | 1 | 0,1 | 1 |
| `solver_consistency_(A≡P≡N)` | equilibrium | anderson,newton,picard | be_secant,be_volume | continuation,fixed | active_set,explicit | 0 | 0 | 1 |
| `storage_secant≡volume` | transient | anderson | be_volume | fixed | active_set | 0 | 0 | 1 |
| `taper_determinism+smooth` | equilibrium | anderson | be_secant | fixed | legacy | 0,1 | 0 | 1,4 |
| `water-budget_closure_(schemes)` | equilibrium | anderson,newton,picard | bdf2_on_V,be_secant,be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit,legacy,off | 1 | 1 | 1 |

## 3. Pairwise crossings

A blank cell is a combination **no run exercises**. `by design` marks the ones WTM refuses on purpose -- those blanks are correct, not gaps.


### solver x collector

| solver \ collector | active_set | explicit | implicit | legacy | off |
|---|---|---|---|---|---|
| **anderson** | 76 | 7 | 17 | 20 | 7 |
| **newton** | 8 | 2 | 2 |   |   |
| **picard** | by design | 3 | 1 | 1 |   |

### integrator x collector

| integrator \ collector | active_set | explicit | implicit | legacy | off |
|---|---|---|---|---|---|
| **bdf2_on_V** | 5 | 2 | 1 | 1 |   |
| **be_secant** | by design | 10 | 16 | 20 | 7 |
| **be_volume** | 67 |   | 2 |   |   |
| **tr_bdf2** | 12 |   | 1 |   |   |

### dtctl x collector

| dtctl \ collector | active_set | explicit | implicit | legacy | off |
|---|---|---|---|---|---|
| **adaptive** | 7 |   | by design |   |   |
| **continuation** | 4 |   | 1 |   |   |
| **fixed** | 73 | 12 | 19 | 21 | 7 |

### run_type x collector

| run_type \ collector | active_set | explicit | implicit | legacy | off |
|---|---|---|---|---|---|
| **equilibrium** | 64 | 12 | 20 | 21 | 7 |
| **test** | 2 |   |   |   |   |
| **transient** | 18 |   |   |   |   |

### solver x integrator

| solver \ integrator | bdf2_on_V | be_secant | be_volume | tr_bdf2 |
|---|---|---|---|---|
| **anderson** | 5 | 48 | 61 | 13 |
| **newton** |   | 4 | 8 |   |
| **picard** | 4 | 1 |   |   |

### run_type x solver

| run_type \ solver | anderson | newton | picard |
|---|---|---|---|
| **equilibrium** | 109 | 10 | 5 |
| **test** | 2 |   |   |
| **transient** | 16 | 2 |   |

### fsm x collector

| fsm \ collector | active_set | explicit | implicit | legacy | off |
|---|---|---|---|---|---|
| **0** | 34 | 7 | 7 | 5 | 6 |
| **1** | 50 | 5 | 13 | 16 | 1 |

### runoff_ratio x dtctl

| runoff_ratio \ dtctl | adaptive | continuation | fixed |
|---|---|---|---|
| **0** | 3 | 3 | 105 |
| **1** | 4 | 2 | 27 |

## 4. Uncovered pairwise crossings

**31** combinations are reachable but exercised by nothing:

- `solver=newton` x `collector=legacy`
- `solver=newton` x `collector=off`
- `solver=picard` x `collector=off`
- `integrator=bdf2_on_V` x `collector=off`
- `integrator=be_volume` x `collector=explicit`
- `integrator=be_volume` x `collector=legacy`
- `integrator=be_volume` x `collector=off`
- `integrator=tr_bdf2` x `collector=explicit`
- `integrator=tr_bdf2` x `collector=legacy`
- `integrator=tr_bdf2` x `collector=off`
- `dtctl=adaptive` x `collector=explicit`
- `dtctl=adaptive` x `collector=legacy`
- `dtctl=adaptive` x `collector=off`
- `dtctl=continuation` x `collector=explicit`
- `dtctl=continuation` x `collector=legacy`
- `dtctl=continuation` x `collector=off`
- `run_type=test` x `collector=explicit`
- `run_type=test` x `collector=implicit`
- `run_type=test` x `collector=legacy`
- `run_type=test` x `collector=off`
- `run_type=transient` x `collector=explicit`
- `run_type=transient` x `collector=implicit`
- `run_type=transient` x `collector=legacy`
- `run_type=transient` x `collector=off`
- `solver=newton` x `integrator=bdf2_on_V`
- `solver=newton` x `integrator=tr_bdf2`
- `solver=picard` x `integrator=be_volume`
- `solver=picard` x `integrator=tr_bdf2`
- `run_type=test` x `solver=newton`
- `run_type=test` x `solver=picard`
- `run_type=transient` x `solver=picard`

Each is a place a defect could live unseen. That is not a demand to cover all of them -- some are uninteresting -- but the list should be read, and anything load-bearing should get an arm.

