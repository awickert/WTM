# WTM test-coverage matrix

**Generated** by `tests/coverage_matrix.py` from the fingerprints WTM itself emits, so it reflects what each run RESOLVED to rather than what its config appears to say. Do not edit by hand; re-run the suite.

Runs recorded: **257** across **34** tests.


## 1. Combination coverage: every (solver, collector, integrator) against run type

The pairwise tables further down CANNOT answer this: two pairs can each be covered while their combination never runs. **0** means that combination has never been run at that run type.

| solver | collector | integrator | `equilibrium` | `test` | `transient` |
|---|---|---|---|---|---|
| `anderson` | `active_set` | `bdf2_on_V` | 10 | **0** | 5 |
| `anderson` | `active_set` | `tr_bdf2` | 92 | 2 | 13 |
| `anderson` | `explicit` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `explicit` | `be_secant` | **0** | **0** | 1 |
| `anderson` | `explicit` | `be_volume` | **0** | **0** | 1 |
| `anderson` | `explicit` | `tr_bdf2` | 13 | **0** | 3 |
| `anderson` | `extended_soil` | `tr_bdf2` | 1 | **0** | **0** |
| `anderson` | `implicit` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `implicit` | `tr_bdf2` | 21 | **0** | 3 |
| `anderson` | `off` | `bdf2_on_V` | 1 | **0** | 1 |
| `anderson` | `off` | `tr_bdf2` | 16 | **0** | 3 |
| `newton` | `active_set` | `bdf2_on_V` | 1 | **0** | 1 |
| `newton` | `active_set` | `be_volume` | 11 | **0** | 4 |
| `newton` | `explicit` | `bdf2_on_V` | 1 | **0** | 1 |
| `newton` | `explicit` | `be_volume` | 4 | **0** | 2 |
| `newton` | `implicit` | `bdf2_on_V` | 1 | **0** | 1 |
| `newton` | `implicit` | `be_volume` | 4 | **0** | 2 |
| `newton` | `off` | `bdf2_on_V` | 1 | **0** | 1 |
| `newton` | `off` | `be_volume` | 2 | **0** | 2 |
| `picard` | `explicit` | `bdf2_on_V` | 3 | **0** | 1 |
| `picard` | `explicit` | `be_volume` | 3 | **0** | 2 |
| `picard` | `implicit` | `bdf2_on_V` | 3 | **0** | 2 |
| `picard` | `implicit` | `be_volume` | 4 | **0** | 4 |
| `picard` | `off` | `bdf2_on_V` | 1 | **0** | 1 |
| `picard` | `off` | `be_volume` | 2 | **0** | 2 |

**25** distinct combinations are exercised at all. Of those, **22** run in BOTH equilibrium and transient, **1** are equilibrium-only and **2** transient-only.

Equilibrium-only, i.e. never exercised on the transient path:

- `anderson` x `extended_soil` x `tr_bdf2`


## 2. What each test covers

| test | run_type | solver | integrator | dtctl | collector | fsm | runoff_ratio | ranks |
|---|---|---|---|---|---|---|---|---|
| `FSM_MPI_consistency` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `FSM_conservation_+_lake` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `MPI_consistency_matrix` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0,1 | 0,1 | 1,4 |
| `Newton_Jacobian_+_contract` | equilibrium | anderson,newton | be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit | 1 | 0 | 1 |
| `active-set_collector-indep` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set,explicit,implicit | 1 | 0 | 1 |
| `adaptive-restart_robustness` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `adaptive_dt_+_water_metric` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `adaptive_estimator_order` | equilibrium | anderson | bdf2_on_V,tr_bdf2 | adaptive | active_set | 0,1 | 1 | 1 |
| `boundary:_analytic_parabola` | equilibrium | anderson | tr_bdf2 | adaptive | off | 0 | 0 | 1 |
| `boundary:_dirichlet≡padding` | equilibrium | anderson,newton | be_volume,tr_bdf2 | adaptive,continuation | explicit | 0 | 0 | 1 |
| `cascade_A->B->ocean_(skim)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `combination_sweep` | equilibrium,transient | anderson,newton,picard | bdf2_on_V,be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit,off | 1 | 0 | 1 |
| `config/flag_route_equality` | equilibrium | newton | be_volume | adaptive,continuation | active_set | 1 | 1 | 1 |
| `dt-sensitivity_(active-set)` | equilibrium | anderson | tr_bdf2 | fixed | active_set,implicit | 0 | 0 | 1 |
| `flicker_1:_storativity_jump` | transient | anderson | bdf2_on_V,tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `flicker_2:_evap_discontinuity` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 0 | 0 | 1 |
| `ghost-boundary_(#96)` | transient | anderson,newton | bdf2_on_V,be_volume,tr_bdf2 | adaptive | active_set | 0 | 0 | 1,4 |
| `ghost-cell_MPI` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1,2 |
| `golden_(expected_results)` | equilibrium,transient | anderson | tr_bdf2 | adaptive | active_set | 0,1 | 0,1 | 1,4 |
| `lake_evap_==_ET_(transition_inert)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `local-in-space_water_ledger` | equilibrium | anderson | tr_bdf2 | adaptive | active_set,off | 0 | 0,1 | 1 |
| `mass-balance_MPI` | test | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `multi-lake_stages_vs_dt` | equilibrium | anderson | tr_bdf2 | fixed | active_set,implicit | 1 | 0 | 4 |
| `nested_DH_+_skim_spill-accuracy` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `recharge_consistency_(#93)` | transient | anderson | bdf2_on_V,tr_bdf2 | fixed | active_set | 0 | 0 | 1 |
| `runoff_collector_selector` | equilibrium | anderson | tr_bdf2 | fixed | active_set,explicit,extended_soil,implicit,off | 0 | 0 | 1 |
| `runoff_gathering_(wtd=0)` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | implicit,off | 0 | 0 | 1 |
| `serial_rank-0_recharge_path` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1,4 |
| `snapshot_name_+_restart` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 0 | 0 | 1 |
| `solve-count_invariance` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0,1 | 1 |
| `solver_consistency_(A≡P≡N)` | equilibrium | anderson,newton,picard | be_volume,tr_bdf2 | adaptive,continuation | active_set,explicit | 0 | 0 | 1 |
| `storage_secant≡volume` | transient | anderson | be_secant,be_volume | adaptive | explicit | 0 | 0 | 1 |
| `taper_determinism+smooth` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set,explicit | 0,1 | 0 | 1,4 |
| `water-budget_closure_(schemes)` | equilibrium | anderson,newton,picard | bdf2_on_V,be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit,off | 1 | 1 | 1 |

## 3. Pairwise crossings

A blank cell is a combination **no run exercises**. `by design` and `does not converge` are LEARNED from tests/combination_sweep, which attempts every combination and records what it did -- so those are observations, not assertions.


### solver x collector

| solver \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **anderson** | 122 | 20 | 1 | 26 | 21 |
| **newton** | 17 | 8 |   | 8 | 6 |
| **picard** | by design | 9 |   | 13 | 6 |

### integrator x collector

| integrator \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **bdf2_on_V** | 17 | 8 |   | 9 | 6 |
| **be_secant** | by design | 1 |   |   |   |
| **be_volume** | 15 | 12 |   | 14 | 8 |
| **tr_bdf2** | 107 | 16 | 1 | 24 | 19 |

### dtctl x collector

| dtctl \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **adaptive** | 96 | 27 |   |   | 26 |
| **continuation** | 11 | 7 |   | 7 | 6 |
| **fixed** | 32 | 3 | 1 | 40 | 1 |

### run_type x collector

| run_type \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **equilibrium** | 114 | 25 | 1 | 34 | 23 |
| **test** | 2 |   |   |   |   |
| **transient** | 23 | 12 |   | 13 | 10 |

### solver x integrator

| solver \ integrator | bdf2_on_V | be_secant | be_volume | tr_bdf2 |
|---|---|---|---|---|
| **anderson** | 21 | 1 | 1 | 167 |
| **newton** | 8 |   | 31 |   |
| **picard** | 11 |   | 17 |   |

### run_type x solver

| run_type \ solver | anderson | newton | picard |
|---|---|---|---|
| **equilibrium** | 156 | 25 | 16 |
| **test** | 2 |   |   |
| **transient** | 32 | 14 | 12 |

### fsm x collector

| fsm \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **0** | 44 | 12 | 1 | 9 | 12 |
| **1** | 95 | 25 |   | 38 | 21 |

### runoff_ratio x dtctl

| runoff_ratio \ dtctl | adaptive | continuation | fixed |
|---|---|---|---|
| **0** | 105 | 28 | 70 |
| **1** | 44 | 3 | 7 |

## 4. Uncovered pairwise crossings

**22** combinations are reachable but exercised by nothing:

- `solver=newton` x `collector=extended_soil`
- `solver=picard` x `collector=extended_soil`
- `integrator=bdf2_on_V` x `collector=extended_soil`
- `integrator=be_secant` x `collector=extended_soil`
- `integrator=be_secant` x `collector=implicit`
- `integrator=be_secant` x `collector=off`
- `integrator=be_volume` x `collector=extended_soil`
- `dtctl=adaptive` x `collector=extended_soil`
- `dtctl=adaptive` x `collector=implicit`
- `dtctl=continuation` x `collector=extended_soil`
- `run_type=test` x `collector=explicit`
- `run_type=test` x `collector=extended_soil`
- `run_type=test` x `collector=implicit`
- `run_type=test` x `collector=off`
- `run_type=transient` x `collector=extended_soil`
- `solver=newton` x `integrator=be_secant`
- `solver=newton` x `integrator=tr_bdf2`
- `solver=picard` x `integrator=be_secant`
- `solver=picard` x `integrator=tr_bdf2`
- `run_type=test` x `solver=newton`
- `run_type=test` x `solver=picard`
- `fsm=1` x `collector=extended_soil`

Each is a place a defect could live unseen. That is not a demand to cover all of them -- some are uninteresting -- but the list should be read, and anything load-bearing should get an arm.

