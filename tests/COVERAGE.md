# WTM test-coverage matrix

**Generated** by `tests/coverage_matrix.py` from the fingerprints WTM itself emits, so it reflects what each run RESOLVED to rather than what its config appears to say. Do not edit by hand; re-run the suite.

Runs recorded: **407** across **200** tests.


## 1. Combination coverage: every (solver, collector, integrator) against run type

The pairwise tables further down CANNOT answer this: two pairs can each be covered while their combination never runs. **0** means that combination has never been run at that run type.

| solver | collector | integrator | `equilibrium` | `test` | `transient` |
|---|---|---|---|---|---|
| `anderson` | `active_set` | `bdf2` | 16 | **0** | 2 |
| `anderson` | `active_set` | `be_volume` | 24 | **0** | 2 |
| `anderson` | `active_set` | `tr_bdf2` | 156 | 2 | 11 |
| `anderson` | `explicit` | `bdf2` | 4 | **0** | 1 |
| `anderson` | `explicit` | `be_secant` | 1 | **0** | 2 |
| `anderson` | `explicit` | `be_volume` | 4 | **0** | 2 |
| `anderson` | `explicit` | `tr_bdf2` | 17 | **0** | 1 |
| `anderson` | `extended_soil` | `tr_bdf2` | 1 | **0** | **0** |
| `anderson` | `implicit` | `bdf2` | 7 | **0** | 1 |
| `anderson` | `implicit` | `be_secant` | 3 | **0** | 1 |
| `anderson` | `implicit` | `be_volume` | 9 | **0** | 1 |
| `anderson` | `implicit` | `tr_bdf2` | 21 | **0** | 1 |
| `anderson` | `off` | `bdf2` | 7 | **0** | 4 |
| `anderson` | `off` | `be_secant` | 1 | **0** | 1 |
| `anderson` | `off` | `be_volume` | 7 | **0** | 5 |
| `anderson` | `off` | `tr_bdf2` | 22 | **0** | 4 |
| `newton` | `active_set` | `bdf2` | 1 | **0** | 1 |
| `newton` | `active_set` | `be_volume` | 10 | **0** | 1 |
| `newton` | `explicit` | `bdf2` | 1 | **0** | 1 |
| `newton` | `explicit` | `be_secant` | 1 | **0** | 1 |
| `newton` | `explicit` | `be_volume` | 3 | **0** | 1 |
| `newton` | `implicit` | `bdf2` | 1 | **0** | 1 |
| `newton` | `implicit` | `be_secant` | 1 | **0** | 1 |
| `newton` | `implicit` | `be_volume` | 4 | **0** | 1 |
| `newton` | `off` | `bdf2` | 1 | **0** | 1 |
| `newton` | `off` | `be_secant` | 1 | **0** | 1 |
| `newton` | `off` | `be_volume` | 1 | **0** | 3 |
| `picard` | `explicit` | `bdf2` | 3 | **0** | 1 |
| `picard` | `explicit` | `be_secant` | 1 | **0** | 1 |
| `picard` | `explicit` | `be_volume` | 2 | **0** | 1 |
| `picard` | `implicit` | `bdf2` | 3 | **0** | 2 |
| `picard` | `implicit` | `be_secant` | 2 | **0** | 2 |
| `picard` | `implicit` | `be_volume` | 2 | **0** | 2 |
| `picard` | `off` | `bdf2` | 1 | **0** | 1 |
| `picard` | `off` | `be_secant` | 1 | **0** | 1 |
| `picard` | `off` | `be_volume` | 1 | **0** | 1 |

**36** distinct combinations are exercised at all. Of those, **35** run in BOTH equilibrium and transient, **1** are equilibrium-only and **0** transient-only.

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
| `adaptive_water/adapt` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `adaptive_water/cc` | equilibrium | anderson | be_volume | fixed | active_set | 0 | 0 | 1 |
| `adaptive_water/water` | equilibrium | anderson | be_volume | fixed | active_set | 0 | 0 | 1 |
| `boundary:_analytic_parabola` | equilibrium | anderson | tr_bdf2 | adaptive | off | 0 | 0 | 1 |
| `boundary:_dirichlet≡padding` | equilibrium | anderson,newton | be_volume,tr_bdf2 | adaptive,continuation | explicit | 0 | 0 | 1 |
| `budget_closure/bdf2v_ad` | equilibrium | anderson | bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `budget_closure/c_as` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `budget_closure/c_ex` | equilibrium | anderson | tr_bdf2 | adaptive | explicit | 1 | 1 | 1 |
| `budget_closure/c_off` | equilibrium | anderson | tr_bdf2 | adaptive | off | 1 | 1 | 1 |
| `budget_closure/c_pex` | equilibrium | picard | bdf2 | adaptive | explicit | 1 | 1 | 1 |
| `budget_closure/c_rof` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `budget_closure/d_and` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `budget_closure/d_nt` | equilibrium | newton | be_volume | continuation | implicit | 1 | 1 | 1 |
| `budget_closure/d_ntc` | equilibrium | newton | be_volume | continuation | implicit | 1 | 1 | 1 |
| `budget_closure/d_ntu` | equilibrium | newton | be_volume | continuation | active_set | 1 | 1 | 1 |
| `budget_closure/d_pic` | equilibrium | picard | bdf2 | adaptive | explicit | 1 | 1 | 1 |
| `budget_closure/f_and` | equilibrium | anderson | be_secant | fixed | implicit | 1 | 1 | 1 |
| `budget_closure/f_vol` | equilibrium | anderson | be_volume | fixed | implicit | 1 | 1 | 1 |
| `budget_closure/s_and` | equilibrium | anderson | be_secant | fixed | implicit | 1 | 1 | 1 |
| `budget_closure/s_pic` | equilibrium | picard | bdf2 | fixed | implicit | 1 | 1 | 1 |
| `budget_closure/s_tr` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 1 | 1 | 1 |
| `budget_closure/s_vol` | equilibrium | anderson | be_volume | fixed | implicit | 1 | 1 | 1 |
| `budget_closure/tr_as_ad` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `cascade_A->B->ocean_(skim)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `combination_sweep/eq_anderson_bdf2v_active_set` | equilibrium | anderson | bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_bdf2v_explicit` | equilibrium | anderson | bdf2 | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_bdf2v_implicit` | equilibrium | anderson | bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_bdf2v_off` | equilibrium | anderson | bdf2 | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_be_explicit` | equilibrium | anderson | be_secant | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_be_implicit` | equilibrium | anderson | be_secant | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_be_off` | equilibrium | anderson | be_secant | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_trbdf2_active_set` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_trbdf2_explicit` | equilibrium | anderson | tr_bdf2 | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_trbdf2_implicit` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_trbdf2_off` | equilibrium | anderson | tr_bdf2 | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_volume_active_set` | equilibrium | anderson | be_volume | adaptive | active_set | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_volume_explicit` | equilibrium | anderson | be_volume | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_volume_implicit` | equilibrium | anderson | be_volume | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_anderson_volume_off` | equilibrium | anderson | be_volume | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/eq_newton_bdf2v_active_set` | equilibrium | newton | bdf2 | continuation | active_set | 1 | 0 | 1 |
| `combination_sweep/eq_newton_bdf2v_explicit` | equilibrium | newton | bdf2 | continuation | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_newton_bdf2v_implicit` | equilibrium | newton | bdf2 | continuation | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_newton_bdf2v_off` | equilibrium | newton | bdf2 | continuation | off | 1 | 0 | 1 |
| `combination_sweep/eq_newton_be_explicit` | equilibrium | newton | be_secant | continuation | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_newton_be_implicit` | equilibrium | newton | be_secant | continuation | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_newton_be_off` | equilibrium | newton | be_secant | continuation | off | 1 | 0 | 1 |
| `combination_sweep/eq_newton_volume_active_set` | equilibrium | newton | be_volume | continuation | active_set | 1 | 0 | 1 |
| `combination_sweep/eq_newton_volume_explicit` | equilibrium | newton | be_volume | continuation | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_newton_volume_implicit` | equilibrium | newton | be_volume | continuation | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_newton_volume_off` | equilibrium | newton | be_volume | continuation | off | 1 | 0 | 1 |
| `combination_sweep/eq_picard_bdf2v_explicit` | equilibrium | picard | bdf2 | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_bdf2v_implicit` | equilibrium | picard | bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_bdf2v_implicit_s` | equilibrium | picard | bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_bdf2v_off` | equilibrium | picard | bdf2 | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/eq_picard_be_explicit` | equilibrium | picard | be_secant | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_be_implicit` | equilibrium | picard | be_secant | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_be_implicit_s` | equilibrium | picard | be_secant | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_be_off` | equilibrium | picard | be_secant | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/eq_picard_volume_explicit` | equilibrium | picard | be_volume | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_volume_implicit` | equilibrium | picard | be_volume | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_volume_implicit_s` | equilibrium | picard | be_volume | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/eq_picard_volume_off` | equilibrium | picard | be_volume | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_bdf2v_active_set` | transient | anderson | bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_bdf2v_explicit` | transient | anderson | bdf2 | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_bdf2v_implicit` | transient | anderson | bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_bdf2v_off` | transient | anderson | bdf2 | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_be_explicit` | transient | anderson | be_secant | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_be_implicit` | transient | anderson | be_secant | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_be_off` | transient | anderson | be_secant | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_trbdf2_active_set` | transient | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_trbdf2_explicit` | transient | anderson | tr_bdf2 | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_trbdf2_implicit` | transient | anderson | tr_bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_trbdf2_off` | transient | anderson | tr_bdf2 | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_volume_active_set` | transient | anderson | be_volume | adaptive | active_set | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_volume_explicit` | transient | anderson | be_volume | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_volume_implicit` | transient | anderson | be_volume | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_anderson_volume_off` | transient | anderson | be_volume | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_newton_bdf2v_active_set` | transient | newton | bdf2 | continuation | active_set | 1 | 0 | 1 |
| `combination_sweep/tr_newton_bdf2v_explicit` | transient | newton | bdf2 | continuation | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_newton_bdf2v_implicit` | transient | newton | bdf2 | continuation | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_newton_bdf2v_off` | transient | newton | bdf2 | continuation | off | 1 | 0 | 1 |
| `combination_sweep/tr_newton_be_explicit` | transient | newton | be_secant | continuation | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_newton_be_implicit` | transient | newton | be_secant | continuation | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_newton_be_off` | transient | newton | be_secant | continuation | off | 1 | 0 | 1 |
| `combination_sweep/tr_newton_volume_active_set` | transient | newton | be_volume | continuation | active_set | 1 | 0 | 1 |
| `combination_sweep/tr_newton_volume_explicit` | transient | newton | be_volume | continuation | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_newton_volume_implicit` | transient | newton | be_volume | continuation | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_newton_volume_off` | transient | newton | be_volume | continuation | off | 1 | 0 | 1 |
| `combination_sweep/tr_picard_bdf2v_explicit` | transient | picard | bdf2 | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_bdf2v_implicit` | transient | picard | bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_bdf2v_implicit_s` | transient | picard | bdf2 | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_bdf2v_off` | transient | picard | bdf2 | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_picard_be_explicit` | transient | picard | be_secant | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_be_implicit` | transient | picard | be_secant | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_be_implicit_s` | transient | picard | be_secant | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_be_off` | transient | picard | be_secant | adaptive | off | 1 | 0 | 1 |
| `combination_sweep/tr_picard_volume_explicit` | transient | picard | be_volume | adaptive | explicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_volume_implicit` | transient | picard | be_volume | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_volume_implicit_s` | transient | picard | be_volume | fixed | implicit | 1 | 0 | 1 |
| `combination_sweep/tr_picard_volume_off` | transient | picard | be_volume | adaptive | off | 1 | 0 | 1 |
| `converged:_active_set` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set,explicit,implicit | 1 | 0 | 1 |
| `converged:_fsm_cascade` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `converged:_fsm_consistency` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,2,4 |
| `converged:_fsm_fullness` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `converged:_lake_evap_equals_et` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `converged:_xrank_adaptive` | transient | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,2,4,6 |
| `converged:_xrank_growth` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1,6 |
| `coupling_convergence/cascade_025yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/cascade_025yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/cascade_05yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/cascade_05yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/cascade_1yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/cascade_1yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/lake_025yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/lake_025yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/lake_05yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/lake_05yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/lake_1yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/lake_1yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/multilake_025yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/multilake_025yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/multilake_05yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/multilake_05yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/multilake_1yr_continuous` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_convergence/multilake_1yr_impulse` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 1 | 0 | 1 |
| `coupling_iteration_(#112)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `cross-rank_adaptive_determinism` | transient | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 4 |
| `cross-rank_drift_regime` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1,6 |
| `dt-sensitivity_(active-set)` | equilibrium | anderson | tr_bdf2 | fixed | active_set,implicit | 0 | 0 | 1 |
| `estimator_order/BDF2_on_V_fsm_off__1971000` | equilibrium | anderson | bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_off__31536000` | equilibrium | anderson | bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_off__492750` | equilibrium | anderson | bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_off__7884000` | equilibrium | anderson | bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_on___1971000` | equilibrium | anderson | bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_on___31536000` | equilibrium | anderson | bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_on___492750` | equilibrium | anderson | bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/BDF2_on_V_fsm_on___7884000` | equilibrium | anderson | bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_off__1971000` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_off__31536000` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_off__492750` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_off__7884000` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_on___1971000` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_on___31536000` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_on___492750` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/TR_BDF2___fsm_on___7884000` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/coarse_TR_BDF2___fsm_on__COARSE__the_range_the_controller_uses___2` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/coarse_TR_BDF2___fsm_on__COARSE__the_range_the_controller_uses___3` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/coarse_TR_BDF2___fsm_on__COARSE__the_range_the_controller_uses___4` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/coarse_TR_BDF2___fsm_on__COARSE__the_range_the_controller_uses___6` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/coarse_TR_BDF2___fsm_on__COARSE__the_range_the_controller_uses___8` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/pre_coarse` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `estimator_order/pre_fine` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `flicker_1:_storativity_jump` | transient | anderson | bdf2,tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `flicker_2:_evap_discontinuity` | equilibrium | anderson | tr_bdf2 | adaptive | off | 0 | 0 | 1 |
| `ghost-boundary_(#96)` | transient | anderson,newton | bdf2,be_volume,tr_bdf2 | adaptive,fixed | active_set,off | 0 | 0 | 1,4 |
| `ghost-cell_MPI` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1,2 |
| `golden_(expected_results)` | equilibrium,transient | anderson | tr_bdf2 | adaptive | active_set | 0,1 | 0,1 | 1,4 |
| `lake_evap_==_ET_(transition_inert)` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1 |
| `local-in-space_water_ledger` | equilibrium | anderson | tr_bdf2 | fixed | active_set,off | 0 | 0,1 | 1 |
| `mass-balance_MPI` | test | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `multi-lake_stages_vs_dt` | equilibrium | anderson | tr_bdf2 | fixed | active_set,implicit | 1 | 0 | 4 |
| `nested_DH_+_skim_spill-accuracy` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 0 | 1,4 |
| `per-step_water_ledger` | equilibrium | anderson | bdf2,be_volume,tr_bdf2 | fixed | active_set,explicit,implicit,off | 1 | 0 | 1 |
| `recharge_consistency_(#93)` | transient | anderson | bdf2,be_volume,tr_bdf2 | fixed | off | 0 | 0 | 1 |
| `route_equality/convergence_metric__v1` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `route_equality/convergence_metric__v2` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `route_equality/fsm_coupling__v1` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `route_equality/fsm_coupling__v2` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `route_equality/newt_alone` | equilibrium | newton | be_volume | continuation | active_set | 1 | 1 | 1 |
| `route_equality/newt_off` | equilibrium | newton | be_volume | fixed | active_set | 1 | 1 | 1 |
| `route_equality/volume_tol__v1` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `route_equality/volume_tol__v2` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1 |
| `runoff_collector/aset` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 0 | 0 | 1 |
| `runoff_collector/explicit` | equilibrium | anderson | tr_bdf2 | fixed | explicit | 0 | 0 | 1 |
| `runoff_collector/implicit` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 0 | 0 | 1 |
| `runoff_collector/off` | equilibrium | anderson | tr_bdf2 | fixed | off | 0 | 0 | 1 |
| `runoff_collector/unset` | equilibrium | anderson | tr_bdf2 | fixed | active_set | 0 | 0 | 1 |
| `runoff_collector/xsoil_mode` | equilibrium | anderson | tr_bdf2 | fixed | extended_soil | 0 | 0 | 1 |
| `runoff_collector_selector` | equilibrium | anderson | tr_bdf2 | fixed | active_set,explicit | 0 | 0 | 1 |
| `runoff_gathering_(wtd=0)` | equilibrium | anderson | tr_bdf2 | fixed | implicit,off | 0 | 0 | 1 |
| `serial_rank-0_recharge_path` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 1 | 1 | 1,4 |
| `snapshot_name_+_restart` | equilibrium | anderson | tr_bdf2 | fixed | implicit | 0 | 0 | 1 |
| `solve-count_invariance` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set | 1 | 0,1 | 1 |
| `solver_consistency/anderson` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `solver_consistency/newton` | equilibrium | newton | be_volume | continuation | active_set | 0 | 0 | 1 |
| `solver_consistency/picard` | equilibrium | picard | be_volume | adaptive | explicit | 0 | 0 | 1 |
| `solver_consistency/volconv` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `solver_consistency/volgov` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `storage_secant≡volume` | transient | anderson | be_secant,be_volume | adaptive | explicit | 0 | 0 | 1 |
| `taper_determinism+smooth` | equilibrium | anderson | tr_bdf2 | adaptive,fixed | active_set,explicit | 0,1 | 0 | 1,4 |
| `tolerance_independence_(#104)` | equilibrium | anderson | be_volume | fixed | active_set | 0 | 0 | 1 |
| `unit:_run-log_header_+_trace_parsing` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |
| `variable_porosity/cc` | equilibrium | anderson | be_volume | adaptive | active_set | 0 | 0 | 1 |
| `variable_porosity/tr` | equilibrium | anderson | tr_bdf2 | adaptive | active_set | 0 | 0 | 1 |

## 3. Pairwise crossings

A blank cell is a combination **no run exercises**. `by design` and `does not converge` are LEARNED from tests/combination_sweep, which attempts every combination and records what it did -- so those are observations, not assertions.


### solver x collector

| solver \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **anderson** | 213 | 32 | 1 | 44 | 51 |
| **newton** | 13 | 8 |   | 9 | 8 |
| **picard** | by design | 9 |   | 13 | 6 |

### integrator x collector

| integrator \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **bdf2** | 20 | 11 |   | 15 | 15 |
| **be_secant** | by design | 7 |   | 10 | 6 |
| **be_volume** | 37 | 13 |   | 19 | 18 |
| **tr_bdf2** | 169 | 18 | 1 | 22 | 26 |

### dtctl x collector

| dtctl \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **adaptive** | 129 | 28 |   |   | 24 |
| **continuation** | 8 | 7 |   | 8 | 6 |
| **fixed** | 89 | 14 | 1 | 58 | 35 |

### run_type x collector

| run_type \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **equilibrium** | 207 | 37 | 1 | 53 | 43 |
| **test** | 2 |   |   |   |   |
| **transient** | 17 | 12 |   | 13 | 22 |

### solver x integrator

| solver \ integrator | bdf2 | be_secant | be_volume | tr_bdf2 |
|---|---|---|---|---|
| **anderson** | 42 | 9 | 54 | 236 |
| **newton** | 8 | 6 | 24 |   |
| **picard** | 11 | 8 | 9 |   |

### run_type x solver

| run_type \ solver | anderson | newton | picard |
|---|---|---|---|
| **equilibrium** | 300 | 25 | 16 |
| **test** | 2 |   |   |
| **transient** | 39 | 13 | 12 |

### fsm x collector

| fsm \ collector | active_set | explicit | extended_soil | implicit | off |
|---|---|---|---|---|---|
| **0** | 54 | 12 | 1 | 7 | 26 |
| **1** | 172 | 37 |   | 59 | 39 |

### runoff_ratio x dtctl

| runoff_ratio \ dtctl | adaptive | continuation | fixed |
|---|---|---|---|
| **0** | 124 | 25 | 186 |
| **1** | 57 | 4 | 11 |

## 4. Uncovered pairwise crossings

**18** combinations are reachable but exercised by nothing:

- `solver=newton` x `collector=extended_soil`
- `solver=picard` x `collector=extended_soil`
- `integrator=bdf2` x `collector=extended_soil`
- `integrator=be_secant` x `collector=extended_soil`
- `integrator=be_volume` x `collector=extended_soil`
- `dtctl=adaptive` x `collector=extended_soil`
- `dtctl=adaptive` x `collector=implicit`
- `dtctl=continuation` x `collector=extended_soil`
- `run_type=test` x `collector=explicit`
- `run_type=test` x `collector=extended_soil`
- `run_type=test` x `collector=implicit`
- `run_type=test` x `collector=off`
- `run_type=transient` x `collector=extended_soil`
- `solver=newton` x `integrator=tr_bdf2`
- `solver=picard` x `integrator=tr_bdf2`
- `run_type=test` x `solver=newton`
- `run_type=test` x `solver=picard`
- `fsm=1` x `collector=extended_soil`

Each is a place a defect could live unseen. That is not a demand to cover all of them -- some are uninteresting -- but the list should be read, and anything load-bearing should get an arm.

