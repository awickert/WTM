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
#   deltat                -> time.deltat
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
if have supplied_wt; then
    case "$(val supplied_wt)" in
        0) echo "  initial_water_table: saturated" ;;
        1) echo "  initial_water_table: supplied" ;;
        *) echo "  initial_water_table: '$(val supplied_wt)'" ;;   # a literal path
    esac
fi

# --- time --------------------------------------------------------------------
if have deltat || have total_time || have report_interval || have save_nreport_interval; then
    echo "time:"
    have deltat                && echo "  deltat: $(val deltat)"
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
    have runoff_collector && echo "  collection:" && echo "    method: $(val runoff_collector)"
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
fi
