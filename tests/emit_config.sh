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
#   cells_per_degree      -> grid.cells_per_degree      (override; fixtures lack georeferencing)
#   southern_edge         -> grid.southern_edge
#   fdepth_a|b|fmin       -> transmissivity.fdepth.a|b|fmin
#   fsm_on 1|0            -> surface_water.mode: routed|ponded
#   runoff_ratio <num>    -> surface_water.runoff_ratio: <num>   (uniform)
#   runoff_ratio_on 1     -> surface_water.runoff_ratio: raster  (require the raster)
#   infiltration_on 0|1   -> surface_water.infiltration_during_flow: false|true
#   runoff_collector      -> surface_water.collection.method
#   extinction_depth      -> evaporation.extinction_depth          (m)
#   under_relaxation      -> dev.under_relaxation
#   et_sigmoid_wtd_center -> evaporation.et_sigmoid.wtd_center     (m)
#   et_sigmoid_width      -> evaporation.et_sigmoid.logistic_width (m)
#   adaptive_dt true|false -> solver.adaptive_dt
#   dt_tol                -> solver.time_step.error_tol
#   dt_max                -> solver.time_step.dt_max
#   t_bar true|false      -> solver.t_bar
#   eq_frac               -> run.equilibrium_stop.frac
#   eq_tol                -> run.equilibrium_stop.tol      (m water; 0 = stop disabled)
#   eq_metric             -> run.equilibrium_stop.metric   (max|rms|frac)
#   land_boundary         -> boundaries.land (dirichlet -> dirichlet_sea_level)
#   storage               -> dev.storage_form (volume | secant)
#   dt_continuation       -> solver.newton.dt_continuation
#   solver_method         -> solver.method (anderson | picard | newton)
#   convergence_metric    -> solver.convergence.metric (head | water)
#   convergence_water_volume_tol -> solver.convergence.water_volume_tol
#   trace                 -> output.trace (a bare list body, e.g. `trace dt, water_step`)
#   time_integration      -> solver.time_integration (backward-euler | bdf2 | tr-bdf2)
#   surfdatadir           -> io.source
#   region|time_start|time_end -> io.region|time_start|time_end
#   textfilename          -> output.run_log
#   outfile_prefix        -> output.outfile_prefix
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

# --- run ---------------------------------------------------------------------
echo "run:"
have run_type && echo "  type: $(val run_type)"
if have eq_frac || have eq_tol || have eq_metric; then
    echo "  equilibrium_stop:"
    have eq_tol    && echo "    tol: $(val eq_tol)"
    have eq_metric && echo "    metric: $(val eq_metric)"
    have eq_frac   && echo "    frac: $(val eq_frac)"
fi
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

# --- grid (override; fixtures are not georeferenced) -------------------------
if have cells_per_degree || have southern_edge; then
    echo "grid:"
    have cells_per_degree && echo "  cells_per_degree: $(val cells_per_degree)"
    have southern_edge    && echo "  southern_edge: $(val southern_edge)"
fi

# --- transmissivity ----------------------------------------------------------
if have fdepth_a || have fdepth_b || have fdepth_fmin; then
    echo "transmissivity:"
    echo "  fdepth:"
    have fdepth_a    && echo "    a: $(val fdepth_a)"
    have fdepth_b    && echo "    b: $(val fdepth_b)"
    have fdepth_fmin && echo "    fmin: $(val fdepth_fmin)"
fi

# --- surface_water -----------------------------------------------------------
if have fsm_on || have runoff_ratio || have runoff_ratio_on || have infiltration_on || have runoff_collector; then
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
    fi
    if have infiltration_on; then
        case "$(val infiltration_on)" in
            1) echo "  infiltration_during_flow: true" ;;
            0) echo "  infiltration_during_flow: false" ;;
        esac
    fi
    if have runoff_collector; then
        echo "  collection:"
        echo "    method: $(val runoff_collector)"
    fi
fi

# --- boundaries --------------------------------------------------------------
# land_boundary was the -wtm_land_boundary flag until it was retired; the config spelling for the
# Dirichlet case is `dirichlet_sea_level`, so translate rather than pass the flag value through.
if have land_boundary; then
    echo "boundaries:"
    case "$(val land_boundary)" in
        dirichlet|dirichlet_sea_level) echo "  land: dirichlet_sea_level" ;;
        *)                             echo "  land: neumann_toposlope" ;;
    esac
fi

# --- solver ---------------------------------------------------------------------
if have adaptive_dt || have dt_tol || have t_bar || have dt_max || have deltat || have dt_continuation || have solver_method || have time_integration; then
    echo "solver:"
    have solver_method && echo "  method: $(val solver_method)"
    have time_integration && echo "  time_integration: $(val time_integration)"
    have adaptive_dt && echo "  adaptive_dt: $(val adaptive_dt)"
    have t_bar       && echo "  t_bar: $(val t_bar)"
    if have dt_continuation; then
        echo "  newton:"
        echo "    dt_continuation: $(val dt_continuation)"
    fi
    if have dt_max || have dt_tol || have deltat; then
        echo "  time_step:"
        have deltat && echo "    dt: $(val deltat)"
        have dt_tol && echo "    error_tol: \"$(val dt_tol)\""
        have dt_max && echo "    dt_max: \"$(val dt_max)\""
    fi
fi

if have convergence_metric || have convergence_water_volume_tol; then
    echo "  convergence:"
    have convergence_metric    && echo "    metric: $(val convergence_metric)"
    have convergence_water_volume_tol && echo "    water_volume_tol: $(val convergence_water_volume_tol)"
fi

# --- dev ---------------------------------------------------------------------
# Both are DEV keys. storage_form exists for tests/storage_equivalence, not for tuning (the two
# assemblies are the same equation -- S is the exact secant). under_relaxation VOIDS a transient
# trajectory: it steps a damped surrogate rather than the problem stated.
if have storage || have under_relaxation; then
    echo "dev:"
    have storage          && echo "  storage_form: $(val storage)"
    have under_relaxation && echo "  under_relaxation: $(val under_relaxation)"
fi

# --- evaporation ---------------------------------------------------------------
if have extinction_depth || have et_sigmoid_wtd_center || have et_sigmoid_width; then
    echo "evaporation:"
    have extinction_depth && echo "  extinction_depth: $(val extinction_depth)"
    if have et_sigmoid_wtd_center || have et_sigmoid_width; then
        echo "  et_sigmoid:"
        have et_sigmoid_wtd_center && echo "    wtd_center: $(val et_sigmoid_wtd_center)"
        have et_sigmoid_width      && echo "    logistic_width: $(val et_sigmoid_width)"
    fi
fi

# --- io ----------------------------------------------------------------------
if have surfdatadir || have region || have time_start || have time_end; then
    echo "io:"
    have surfdatadir && echo "  source: '$(val surfdatadir)'"
    have region      && echo "  region: '$(val region)'"
    have time_start  && echo "  time_start: '$(val time_start)'"
    have time_end    && echo "  time_end: '$(val time_end)'"
fi

# --- output ------------------------------------------------------------------
if have textfilename || have outfile_prefix; then
    echo "output:"
    have outfile_prefix && echo "  outfile_prefix: '$(val outfile_prefix)'"
    have textfilename   && echo "  run_log: '$(val textfilename)'"
    have trace          && echo "  trace: [$(val trace)]"
fi

# The script's exit status is its LAST command's, and every emitter here is a `have X && echo ...`
# list that returns 1 when X is unset. Adding a conditional key at the END of the last block therefore
# made emit_config.sh exit 1 on any config that did not set it -- which broke four MPI suites that call
# it through subprocess.run(check=True), with no clue at the call site. Terminate explicitly so the
# status reflects a real failure (set -e already handles those) rather than the last key's presence.
:
