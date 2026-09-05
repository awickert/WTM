# WTM test-coverage matrix

**Generated** by `tests/coverage_matrix.py` from the fingerprints WTM itself emits, so it reflects what each run RESOLVED to rather than what its config appears to say. Do not edit by hand; re-run the suite.

Runs recorded: **347** across **37** tests.


## 1. Combination coverage: every (solver, collector, integrator) against run type

The pairwise tables further down CANNOT answer this: two pairs can each be covered while their combination never runs. **0** means that combination has never been run at that run type.

| solver | collector | integrator | `equilibrium` | `test` | `transient` |
|---|---|---|---|---|---|
| `anderson` | `active_set` | `bdf2` | 16 | **0** | 5 |
| `anderson` | `active_set` | `be_volume` | 6 | **0** | **0** |
| `anderson` | `active_set` | `tr_bdf2` | 122 | 2 | 15 |
| `anderson` | `explicit` | `bdf2` | 4 | **0** | 1 |
| `anderson` | `explicit` | `be_secant` | **0** | **0** | 1 |
| `anderson` | `explicit` | `be_volume` | 3 | **0** | 1 |
| `anderson` | `explicit` | `tr_bdf2` | 16 | **0** | 3 |
| `anderson` | `extended_soil` | `tr_bdf2` | 1 | **0** | **0** |
| `anderson` | `implicit` | `bdf2` | 7 | **0** | 1 |
| `anderson` | `implicit` | `be_volume` | 6 | **0** | **0** |
| `anderson` | `implicit` | `tr_bdf2` | 27 | **0** | 3 |
| `anderson` | `off` | `bdf2` | 7 | **0** | 1 |
| `anderson` | `off` | `be_volume` | 6 | **0** | **0** |
| `anderson` | `off` | `tr_bdf2` | 22 | **0** | 3 |
| `newton` | `active_set` | `bdf2` | 1 | **0** | 1 |
| `newton` | `active_set` | `be_volume` | 11 | **0** | 4 |
| `newton` | `explicit` | `bdf2` | 1 | **0** | 1 |
| `newton` | `explicit` | `be_volume` | 4 | **0** | 2 |
| `newton` | `implicit` | `bdf2` | 1 | **0** | 1 |
| `newton` | `implicit` | `be_volume` | 5 | **0** | 2 |
| `newton` | `off` | `bdf2` | 1 | **0** | 1 |
| `newton` | `off` | `be_volume` | 2 | **0** | 2 |
| `picard` | `explicit` | `bdf2` | 3 | **0** | 1 |
| `picard` | `explicit` | `be_volume` | 3 | **0** | 2 |
| `picard` | `implicit` | `bdf2` | 3 | **0** | 2 |
| `picard` | `implicit` | `be_volume` | 4 | **0** | 4 |
| `picard` | `off` | `bdf2` | 1 | **0** | 1 |
| `picard` | `off` | `be_volume` | 2 | **0** | 2 |

**28** distinct combinations are exercised at all. Of those, **23** run in BOTH equilibrium and transient, **4** are equilibrium-only and **1** transient-only.

Equilibrium-only, i.e. never exercised on the transient path:

- `anderson` x `active_set` x `be_volume`
- `anderson` x `extended_soil` x `tr_bdf2`
- `anderson` x `implicit` x `be_volume`
- `anderson` x `off` x `be_volume`


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
| `adaptive_estimator_order` | equilibrium | anderson | bdf2,tr_bdf2 | adaptive | active_set | 0,1 | 1 | 1 |
| `boundary:_analytic_parabola` | equilibrium | anderson | tr_bdf2 | adaptive | off | 0 | 0 | 1 |
| `boundary:_dirichlet≡padding` | equilibrium | anderson,newton | be_volume,tr_bdf2 | adaptive,continuation | explicit | 0 | 0 | 1 |
| `cascade_A->B->ocean_(skim)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `combination_sweep` | equilibrium,transient | anderson,newton,picard | bdf2,be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit,off | 1 | 0 | 1 |
| `config/flag_route_equality` | equilibrium | newton | be_volume | adaptive,continuation | active_set | 1 | 1 | 1 |
| `coupling_convergence` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `cross-rank_drift_regime` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1,6 |
| `dt-sensitivity_(active-set)` | equilibrium | anderson | tr_bdf2 | fixed | active_set,implicit | 0 | 0 | 1 |
| `flicker_1:_storativity_jump` | transient | anderson | bdf2,tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `flicker_2:_evap_discontinuity` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 0 | 0 | 1 |
| `ghost-boundary_(#96)` | transient | anderson,newton | bdf2,be_volume,tr_bdf2 | adaptive | active_set | 0 | 0 | 1,4 |
| `ghost-cell_MPI` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1,2 |
| `golden_(expected_results)` | equilibrium,transient | anderson | tr_bdf2 | adaptive | active_set | 0,1 | 0,1 | 1,4 |
| `lake_evap_==_ET_(transition_inert)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `local-in-space_water_ledger` | equilibrium | anderson | tr_bdf2 | adaptive | active_set,off | 0 | 0,1 | 1 |
| `mass-balance_MPI` | test | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `multi-lake_stages_vs_dt` | equilibrium | anderson | tr_bdf2 | fixed | active_set,implicit | 1 | 0 | 4 |
| `nested_DH_+_skim_spill-accuracy` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `per-step_water_ledger` | equilibrium | anderson | bdf2,be_volume,tr_bdf2 | fixed | active_set,explicit,implicit,off | 1 | 0 | 1 |
| `recharge_consistency_(#93)` | transient | anderson | bdf2,tr_bdf2 | fixed | active_set | 0 | 0 | 1 |
| `runoff_collector_selector` | equilibrium | anderson | tr_bdf2 | fixed | active_set,explicit,extended_soil,implicit,off | 0 | 0 | 1 |
| `runoff_gathering_(wtd=0)` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | implicit,off | 0 | 0 | 1 |
| `serial_rank-0_recharge_path` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1,4 |
| `snapshot_name_+_restart` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 0 | 0 | 1 |
| `solve-count_invariance` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set | 1 | 0,1 | 1 |
| `solver_consistency_(A≡P≡N)` | equilibrium | anderson,newton,picard | be_volume,tr_bdf2 | adaptive,continuation | active_set,explicit | 0 | 0 | 1 |
| `storage_secant≡volume` | transient | anderson | be_secant,be_volume | adaptive | explicit | 0 | 0 | 1 |
| `taper_determinism+smooth` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set,explicit | 0,1 | 0 | 1,4 |
| `water-budget_closure_(schemes)` | equilibrium | anderson,newton,picard | bdf2,be_volume,tr_bdf2 | adaptive,continuation,fixed | active_set,explicit,implicit,off | 1 | 1 | 1 |

## 3. Pairwise crossings

A blank cell is a combination **no run exercises**. `by design` and `does not converge` are LEARNED from tests/combination_sweep, which attempts every combination and records what it did -- so those are observations, not assertions.


### solver x collector

| solver \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **anderson** | 166 | 29 | 1 | 44 | 39 |
| **newton** | 17 | 8 |   | 9 | 6 |
| **picard** | by design | 9 |   | 13 | 6 |

### integrator x collector

| integrator \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **bdf2** | 23 | 11 |   | 15 | 12 |
| **be_secant** | by design | 1 |   |   |   |
| **be_volume** | 21 | 15 |   | 21 | 14 |
| **tr_bdf2** | 139 | 19 | 1 | 30 | 25 |

### dtctl x collector

| dtctl \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **adaptive** | 102 | 27 |   |   | 26 |
| **continuation** | 10 | 7 |   | 8 | 6 |
| **fixed** | 71 | 12 | 1 | 58 | 19 |

### run_type x collector

| run_type \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **equilibrium** | 156 | 34 | 1 | 53 | 41 |
| **test** | 2 |   |   |   |   |
| **transient** | 25 | 12 |   | 13 | 10 |

### solver x integrator

| solver \ integrator | bdf2 | be_secant | be_volume | tr_bdf2 |
|---|---|---|---|---|
| **anderson** | 42 | 1 | 22 | 214 |
| **newton** | 8 |   | 32 |   |
| **picard** | 11 |   | 17 |   |

### run_type x solver

| run_type \ solver | anderson | newton | picard |
|---|---|---|---|
| **equilibrium** | 243 | 26 | 16 |
| **test** | 2 |   |   |
| **transient** | 34 | 14 | 12 |

### fsm x collector

| fsm \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **0** | 46 | 12 | 1 | 9 | 12 |
| **1** | 137 | 34 |   | 57 | 39 |

### runoff_ratio x dtctl

| runoff_ratio \ dtctl | adaptive | continuation | fixed |
|---|---|---|---|
| **0** | 108 | 27 | 153 |
| **1** | 47 | 4 | 8 |

## 4. Uncovered pairwise crossings

**22** combinations are reachable but exercised by nothing:

- `solver=newton` x `collector=extended_soil`
- `solver=picard` x `collector=extended_soil`
- `integrator=bdf2` x `collector=extended_soil`
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

