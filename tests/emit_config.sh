#!/usr/bin/env bash
# Shared config emitter for the test suite: read legacy "key value" lines on stdin,
# write the equivalent nested-YAML config (config.yaml schema) on stdout.
#
#   base_cfg | overrides... | tests/emit_config.sh > run.yaml
#
# WHY a translation shim (not YAML heredocs in every test): the config SCHEMA is still
# settling (surface_water.collection is parked on the active-set work #100; the grid: block
# retires when geometry derives from the GDAL geotransform #124; dt_tol renames in #109).
# Centralising the legacy->YAML mapping HERE means a schema change touches one file, not the
# ~20 test runners. It also makes this migration low-risk: every test keeps its exact keys and
# values -- only the serialisation changes. When the schema freezes, retiring the legacy
# vocabulary from the test heredocs is a separate, mechanical pass.
#
# Legacy key -> YAML path map (the full vocabulary the suite uses):
#   run_type              -> run.type
#   supplied_wt 0|1       -> run.initial_water_table: saturated|supplied  (a <path> value passes through)
#   deltat                -> solver.time_step.dt
#   total_time            -> time.total
#   report_interval       -> time.report_interval
#   save_nreport_interval -> time.save_every_n_reports
#   fdepth_a|b|fmin       -> transmissivity.fdepth.a|b|fmin
#   fsm_on 1|0            -> surface_water.mode: routed|ponded
#   runoff_ratio <num>    -> surface_water.runoff_ratio: <num>   (uniform)
#   runoff_ratio_on 1     -> surface_water.runoff_ratio: raster  (require the raster)
#   infiltration_on 0|1   -> surface_water.infiltration_during_flow: false|true
#   runoff_collector      -> surface_water.collection.method
#   fsm_coupling          -> surface_water.fsm_coupling (impulse | continuous)
#   extinction_depth      -> evaporation.extinction_depth
#   taper_surface_transition -> evaporation.tapers.surface_transition  (was -wtm_evap_taper)
#   taper_depth_extinction   -> evaporation.tapers.depth_extinction    (was -wtm_extinction)          (m)
#   under_relaxation      -> dev.under_relaxation
#   et_sigmoid_wtd_center -> evaporation.et_sigmoid.wtd_center     (m)
#   et_sigmoid_width      -> evaporation.et_sigmoid.logistic_width (m)
#   time_step_mode        -> solver.time_step.mode (fixed | adaptive | ramp)
#   dt_tol                -> solver.time_step.error_tol
#   dt_max                -> solver.time_step.dt_max
#   t_bar true|false      -> solver.t_bar
#   eq_frac               -> run.equilibrium_stop.frac
#   eq_tol                -> run.equilibrium_stop.tol      (m water; 0 = stop disabled)
#   eq_metric             -> run.equilibrium_stop.metric   (max|rms|frac)
#   land_boundary         -> boundaries.land (dirichlet -> dirichlet_sea_level)
#   storage               -> dev.storage_form (volume | secant)
#   solver_method         -> solver.method (anderson | picard | newton)
#   convergence_metric    -> solver.convergence.metric (head | volume)
#   convergence_water_volume_tol -> solver.convergence.water_volume_tol
#   trace                 -> output.trace (a bare list body, e.g. `trace dt, water_step`)
#   time_integration      -> solver.time_integration (backward-euler | bdf2 | tr-bdf2)
#   surfdatadir           -> io.source
#   region|time_start|time_end -> io.region|time_start|time_end
#   textfilename          -> output.run_log
#   outfile_prefix        -> output.outfile_prefix
#   run_dir               -> output.directory (+ output.if_exists: overwrite). OPTIONAL: when unset it
#                            is DERIVED as '<outfile_prefix>prov' so every run records provenance.
#   evap_mode             -> DROPPED. No longer a config key; the member is frozen at 0 and is inert
#                            under the default evaporation taper (taper-first: evap_mode is only
#                            consulted with -wtm_evap_taper OFF). A test that needs the legacy
#                            hard-switch must pass -wtm_evap_taper 0 on the CLI, not set this.
set -euo pipefail

declare -A V
while IFS= read -r line; do
    line="${line%%#*}"                    # strip trailing comments
    key="${line%%[[:space:]]*}"           # first token
    [[ -z "$key" ]] && continue           # blank line
    rest="${line#"$key"}"                 # everything after the key
    rest="${rest#"${rest%%[![:space:]]*}"}"   # ltrim
    rest="${rest%"${rest##*[![:space:]]}"}"   # rtrim
    V["$key"]="$rest"
done

have() { [[ -n "${V[$1]+x}" ]]; }
val()  { printf '%s' "${V[$1]}"; }

# EMIT WITH A DEFAULT. The value a suite set, or the MODEL's default when it set none.
#
# WHY EVERY KEY IS EMITTED RATHER THAN OMITTED. A test config must state every setting its run
# resolves to (tests/config_identity.py): what was tested should be exactly what was written down.
# Omitting a key and letting the model default it is how four suites came to measure something other
# than their arm names claimed -- an omitted key meant "the old default" when the tests were written
# and means `auto` now, and nothing re-read the tests when that changed.
#
# The defaults below are the MODEL'S OWN, taken from a minimal run's full_config.yaml, so emitting
# them cannot change any answer: setting a key to the value it would have taken anyway is a no-op.
# That is the property that makes this safe to do in one sweep rather than suite by suite.
def_()  { if have "$1"; then val "$1"; else printf '%s' "$2"; fi; }

# REFUSE A KEY THIS SHIM DOES NOT CONSUME. Until now an unrecognised legacy key was silently
# DROPPED, and this sits UPSTREAM of every guard the model has: WTM aborts on an unknown YAML key
# and on an unread -wtm_ flag, but neither ever sees a key the shim swallowed. That is not
# hypothetical -- it is how a set of collector arms went vacuous, every one of them running the
# default because the tests said `collection_method` while the shim only knows `runoff_collector`.
# The arms passed, and what they proved was nothing. The cost of that failure is not a lost setting,
# it is a LOST NEGATIVE RESULT.
#
# The vocabulary is derived from THIS SCRIPT'S OWN have/val calls rather than kept as a hand-written
# list beside them, so it cannot drift from the code that consumes it. (Safe because every call site
# names a literal key; there are no dynamic lookups.)
mapfile -t KNOWN < <(grep -vE '^[[:space:]]*#' "$0" | grep -oE '\b(have|val|def_) [a-z_0-9]+' \
                     | awk '{print $2}' | sort -u)
# Keys the shim ACCEPTS AND DELIBERATELY IGNORES. Each needs a reason in the header map above, and
# each is announced on stderr rather than swallowed -- a test that sets one should see that it did
# nothing, which is the whole point of this guard.
ACCEPTED_INERT=(evap_mode)
declare -A IS_KNOWN=(); for k in "${KNOWN[@]}" "${ACCEPTED_INERT[@]}"; do IS_KNOWN["$k"]=1; done
for k in "${ACCEPTED_INERT[@]}"; do
    have "$k" && printf 'emit_config.sh: note: %s is accepted but INERT (see the key map above); it sets nothing.\n' "$k" >&2
done
unknown=()
for k in "${!V[@]}"; do [[ -n "${IS_KNOWN[$k]+x}" ]] || unknown+=("$k"); done
if (( ${#unknown[@]} )); then
    printf 'emit_config.sh: unknown key(s), refusing to emit a config that silently drops them:\n' >&2
    for k in "${unknown[@]}"; do
        printf '  %s\n' "$k" >&2
        # did-you-mean: same first token, or a short edit distance by shared prefix
        for c in "${KNOWN[@]}"; do
            [[ "${c%%_*}" == "${k%%_*}" || "$c" == *"${k#*_}"* ]] && printf '      did you mean: %s ?\n' "$c" >&2
        done
    done
    printf '  known keys: %s\n' "${KNOWN[*]}" >&2
    exit 2
fi

# ---- RESOLVE FIRST, EMIT SECOND ---------------------------------------------------------------
# Only two things are resolved up here, and only because surface_water is emitted before solver:
# $_ctl gates the step-controller dials and must exist before the first section is written.
#
# THE MIRROR THAT LIVED HERE IS GONE (2026-09-09, Andy: "the mirror undercuts much of the point of
# requiring explicit inputs from the tests"). It re-derived in bash what the model derives in C++ --
# collection.method, time_integration, time_step.mode, error_tol -- so that every
# generated config could state them. Two things were wrong with it, and the second cost real work:
#
#   IT LAUNDERED DEFAULTS INTO INTENTIONS. Nobody hand-writes these configs; this script generates
#   them. So a value picked by a bash fallback arrived in the author's voice, and a default nobody had
#   considered became indistinguishable from a deliberate setting -- inside the one mechanism whose
#   whole purpose is to stop a config lying about its run. An omitted key is HONEST: it says the author
#   did not choose. A written one says they did.
#
#   THE COPY WAS AN INCOMPLETE MODEL OF THE REAL ONE. It knew the defaults but not the COMBINATION
#   rules. With infiltration_during_flow: true the model resolves an unset fsm_coupling to `impulse`
#   and refuses an explicit `continuous` by name (src/transient_groundwater.cpp:1493); the mirror wrote
#   `continuous` -- the default it knew -- and tests/serial_recharge stopped running. That was then
#   reported as "these keys cannot be declared", when what could not be declared was a guess.
#
# A DERIVED KEY IS NOW EMITTED ONLY IF THE SUITE STATES IT. Until it does, tests/config_identity.py
# reports it MISSING, which is the honest and visible state: the run resolved a setting the test never
# chose. The fix is one line in the suite, next to the arm it belongs to.
_method=$(have solver_method && val solver_method || echo anderson)   # a CONSTANT default, not derived
# A controller runs under `adaptive` or `ramp`, not under `fixed`. This reads only what the SUITE
# STATED -- an unstated mode leaves _ctl false and the dials undeclared, which config_identity.py then
# reports as MISSING. That is the intended visible state, not a gap to paper over.
_mode=$(have time_step_mode && val time_step_mode || echo "")
_ctl=false
[[ "$_mode" == adaptive || "$_mode" == ramp ]] && _ctl=true


# --- run ---------------------------------------------------------------------
echo "run:"
have run_type && echo "  type: $(val run_type)"
echo "  equilibrium_stop:"
# tol has no constant default either: CreateSNES gives an equilibrium run 0.001 and a transient run
# 0.0, and the difference is not cosmetic -- `tol: 0` on an equilibrium run flips the adaptive step
# tolerance from min(eq_tol,0.5)=0.001 to the never-stop branch's 0.5, which moved tests/xrank_growth.
# A run that stops on this tolerance says what it stops at.
have eq_tol && echo "    tol: $(val eq_tol)"
echo "    metric: $(def_ eq_metric frac)"
echo "    frac: $(def_ eq_frac 0.001)"
if have supplied_wt; then
    case "$(val supplied_wt)" in
        0) echo "  initial_water_table: saturated" ;;
        1) echo "  initial_water_table: supplied" ;;
        *) echo "  initial_water_table: '$(val supplied_wt)'" ;;   # a literal path
    esac
fi

# --- time --------------------------------------------------------------------
if have total_time || have report_interval || have save_nreport_interval; then
    echo "time:"
    have total_time            && echo "  total: \"$(val total_time)\""
    have report_interval       && echo "  report_interval: \"$(val report_interval)\""
    have save_nreport_interval && echo "  save_every_n_reports: $(val save_nreport_interval)"
fi

# --- grid: REMOVED. Geometry comes from the input raster's GDAL geotransform (#124), full stop.
# The `grid:` block was a deprecated override that the model READ, WARNED about, and then IGNORED
# whenever a geotransform was present -- which was 629 of 685 measured runs, while 0 used the
# no-geotransform fallback. A key that reads as a choice and is not one is the defect this shim
# exists to prevent, so it is gone rather than merely discouraged. The resolved geometry is recorded
# in full_config.yaml under `derived:`, where it belongs.

# --- transmissivity ----------------------------------------------------------
if have fdepth_a || have fdepth_b || have fdepth_fmin; then
    echo "transmissivity:"
    echo "  fdepth:"
    have fdepth_a    && echo "    a: $(val fdepth_a)"
    have fdepth_b    && echo "    b: $(val fdepth_b)"
    have fdepth_fmin && echo "    fmin: $(val fdepth_fmin)"
    echo "  additive_background_transmissivity: $(def_ t_bedrock 0)"
fi

# --- surface_water -----------------------------------------------------------
if true; then   # always: collection.method and fsm_coupling are resolved for every run
    echo "surface_water:"
    if have fsm_on; then
        case "$(val fsm_on)" in
            1) echo "  mode: routed" ;;
            0) echo "  mode: ponded" ;;
        esac
    fi
    # runoff_ratio: a numeric value (uniform) takes precedence; else runoff_ratio_on 1 requires the raster.
    if have runoff_ratio; then
        echo "  runoff_ratio: $(val runoff_ratio)"
    elif [[ "$(have runoff_ratio_on && val runoff_ratio_on)" == "1" ]]; then
        echo "  runoff_ratio: raster"
    else
        echo "  runoff_ratio: 0"
    fi
    if have infiltration_on; then
        case "$(val infiltration_on)" in
            1) echo "  infiltration_during_flow: true" ;;
            0) echo "  infiltration_during_flow: false" ;;
        esac
    fi
    # fsm_coupling and collection.method are NOT mirrored, and this is the boundary of what mirroring
    # can do. For both, SETTING the key to its resolved value is not the same input as leaving it unset:
    #
    #   fsm_coupling        with infiltration_during_flow: true the model RESOLVES an unset key to
    #                       impulse, but REFUSES an explicit `continuous` by name (#49). Mirroring
    #                       `continuous` turned tests/serial_recharge from a working run into a hard
    #                       error -- a valid configuration made invalid by writing down its own answer.
    #   collection.method   an unset key is DOWNGRADED active_set -> explicit on the Picard path, and
    #                       the model announces it. Setting it suppresses both the downgrade and the
    #                       NOTE, which is precisely what tests/budget_closure's d_pic arm exists to
    #                       observe. Mirroring deleted its subject.
    #
    # These resolutions depend on COMBINATIONS (collector x infiltration x method x run_type), and a
    # bash copy of that is the "do not re-derive the model's policy" rule of #24, which I wrote and then
    # broke. The keys stay undeclared until the model itself can be asked for its resolved config.
    have fsm_coupling && echo "  fsm_coupling: $(val fsm_coupling)"
    if have runoff_collector; then
        echo "  collection:"
        echo "    method: $(val runoff_collector)"
    fi
fi

# --- boundaries --------------------------------------------------------------
# land_boundary was the -wtm_land_boundary flag until it was retired; the config spelling for the
# Dirichlet case is `dirichlet_sea_level`, so translate rather than pass the flag value through.
echo "boundaries:"
case "$(def_ land_boundary neumann_toposlope)" in
    dirichlet|dirichlet_sea_level) echo "  land: dirichlet_sea_level" ;;
    *)                             echo "  land: neumann_toposlope" ;;
esac

# --- solver ---------------------------------------------------------------------
echo "solver:"
echo "  method: $_method"
have time_integration && echo "  time_integration: $(val time_integration)"
echo "  tolerance: $(def_ snes_stol 1e-8)"
echo "  max_iterations: $(def_ max_iterations 10000)"
echo "  t_bar: $(def_ t_bar false)"
echo "  convergence:"
echo "    metric: $(def_ convergence_metric volume)"
echo "    water_volume_tol: $(def_ convergence_water_volume_tol 1e-08)"
echo "  smoothing:"
echo "    ksat_surface: $(def_ ksat_surface_smoothing 0)"
echo "    ksat_soilbottom: $(def_ ksat_soilbottom_smoothing 0)"
echo "    storativity_surface: $(def_ storativity_surface_smoothing 0.01)"
echo "  anderson:"
echo "    restart:"
echo "      enabled: $(def_ ar_enabled false)"
echo "      rho: $(def_ ar_rho 0.9)"
echo "      patience: $(def_ ar_patience 2)"
echo "      max_it: $(def_ ar_max_it 40)"
echo "      max_restarts: $(def_ ar_max_restarts 30)"
echo "  time_step:"
[[ -n "$_mode" ]] && echo "    mode: $_mode"
have deltat && echo "    dt: $(val deltat)"
# error_tol: the adaptive step tolerance. There is no constant default -- the model TRACKS the
# equilibrium-stop tolerance, capped at the free-surface ring bound (CreateSNES.cpp):
#     equilibrium && eq_tol > 0 -> min(eq_tol, 0.5)
#     equilibrium               -> 0.5   (a never-stopping run: pure accuracy knob)
#     transient                 -> 0.1   (no stop criterion to track)
# A suite whose steps are sized by that tolerance states which of the three it means.
have dt_tol && echo "    error_tol: \"$(val dt_tol)\""
have dt_max && echo "    dt_max: \"$(val dt_max)\""
# THE STEP-CONTROLLER DIALS, only when a controller actually runs. They bridge to -wtm_dtc_* flags that
# nothing parses on a fixed-step run, and the model ABORTS on a flag nothing read -- rightly: a dial on a
# controller that is not running is not a setting of the run. full_config.yaml emits them under the same
# condition (src/WTM.cpp), so the two agree and config_identity has nothing to report either way.
if [[ "$_ctl" == true ]]; then
    echo "    grow: $(def_ dtc_grow 1.5)"
    echo "    shrink: $(def_ dtc_shrink 0.25)"
    echo "    grow_if_niter_leq: $(def_ dtc_easy_iters 8)"
    echo "    max_retries: $(def_ dtc_max_retries 15)"
    # norm narrower still -- parsed in the ADAPTIVE branch only, not Newton's ramp.
    [[ "$_mode" == adaptive ]] && echo "    norm: $(def_ dt_norm rms)"
fi

# --- dev ---------------------------------------------------------------------
# Both are DEV keys. storage_form exists for tests/storage_equivalence, not for tuning (the two
# assemblies are the same equation -- S is the exact secant). under_relaxation VOIDS a transient
# trajectory: it steps a damped surrogate rather than the problem stated.
echo "dev:"
echo "  allow_aboveground_water_columns: $(def_ allow_aboveground false)"
echo "  storage_form: $(def_ storage volume)"
echo "  under_relaxation: $(def_ under_relaxation 1)"

echo "parallel:"
echo "  threads_per_rank: $(def_ threads_per_rank 1)"

# --- evaporation ---------------------------------------------------------------
echo "evaporation:"
echo "  et_sigmoid:"
echo "    wtd_center: $(def_ et_sigmoid_wtd_center 0.05)"
echo "    logistic_width: $(def_ et_sigmoid_width 0.1)"
echo "  extinction_depth: $(def_ extinction_depth 8)"
echo "  tapers:"
echo "    surface_transition: $(def_ taper_surface_transition true)"
echo "    depth_extinction: $(def_ taper_depth_extinction true)"

# --- io ----------------------------------------------------------------------
if have surfdatadir || have region || have time_start || have time_end; then
    echo "io:"
    have surfdatadir && echo "  source: '$(val surfdatadir)'"
    have region      && echo "  region: '$(val region)'"
    have time_start  && echo "  time_start: '$(val time_start)'"
    have time_end    && echo "  time_end: '$(val time_end)'"
fi

# --- output ------------------------------------------------------------------
# `trace` joins the gate rather than hiding behind it: it was emitted only when a PATH key was also
# present, so a config asking for `trace` and nothing else silently got no trace channel at all -- the
# same shape of failure as an unknown key, and invisible for the same reason.
if have textfilename || have outfile_prefix || have trace || have run_dir; then
    echo "output:"
    have outfile_prefix && echo "  outfile_prefix: '$(val outfile_prefix)'"
    have textfilename   && echo "  run_log: '$(val textfilename)'"
    echo "  trace: [$(def_ trace '')]"
    echo "  verbosity: $(def_ verbosity normal)"
    # EVERY RUN GETS A PROVENANCE RECORD. output.directory is what gates write_provenance() and
    # write_full_config() in the model (src/WTM.cpp), and no test had ever set it -- so not one run in
    # the suite recorded which binary produced it. That is exactly how a measurement got attributed to
    # the wrong build and published (see the CORRECTION in bce7cc8): had each compared run carried its
    # own commit hash on disk, the mistake would have been visible immediately.
    #
    # Derived from outfile_prefix rather than asked of all 37 runners, so it cannot be forgotten in a
    # new test: `<prefix>prov/`, beside the outputs it describes, one directory per ARM (the prefix is
    # already per-arm) so arms cannot overwrite each other's record. `if_exists: overwrite` keeps the
    # path deterministic -- the default `increment` would mint run<NNN>_<timestamp>/ every time and the
    # directory would be unfindable from the test.
    #
    # This costs no path churn, which is worth stating because it was expected to: outfile_prefix and
    # run_log are rewritten as `run_dir / prefix`, and std::filesystem replaces rather than appends when
    # the right-hand side is ABSOLUTE. Every test passes an absolute prefix, so the outputs stay exactly
    # where they were. VERIFIED by running it before wiring it in.
    if have run_dir; then
        echo "  directory: '$(val run_dir)'"
        echo "  if_exists: overwrite"
    elif have outfile_prefix; then
        echo "  directory: '$(val outfile_prefix)prov'"
        echo "  if_exists: overwrite"
    fi
fi

# The script's exit status is its LAST command's, and every emitter here is a `have X && echo ...`
# list that returns 1 when X is unset. Adding a conditional key at the END of the last block therefore
# made emit_config.sh exit 1 on any config that did not set it -- which broke four MPI suites that call
# it through subprocess.run(check=True), with no clue at the call site. Terminate explicitly so the
# status reflects a real failure (set -e already handles those) rather than the last key's presence.
:
