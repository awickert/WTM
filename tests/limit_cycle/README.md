# `limit_cycle` — a "weird golden": the flicker is an *expected wrong* solution, kept as a diagnostic

**Read this before you touch this test.**

This test deliberately reproduces a **known-wrong, non-physical result** — the *lakeshore flicker* — and
asserts that it is present. That is intentional. The flicker is **not** a target and **not** something to
"fix" by making it disappear here.

## What the flicker is

Where the water table sits at the land surface (`wtd = 0`), the backward-Euler + Anderson solve
**overshoots** the free boundary each step: the storativity jumps (porosity below → ~1 above) and the
exfiltration/exfiltration term switches on. Recharge pushes the table up across the surface, exfiltration pulls
it back, and it repeats — a period-2 **limit cycle**. The per-cycle `|Δwtd|` never decays; the "solution"
oscillates forever. It is a numerical artifact, not physics.

## Why we keep it — the diagnostic

Because the overshoot is unmanaged, **different volume-resolving time-integration schemes** (backward-Euler
`cc`, BDF2-on-V, TR-BDF2, …) settle into *different* flickering wrong states. So running more than one
method is itself an **overshoot detector**: when they disagree at the free boundary, an unmanaged surface
is the reason. This test captures that so the diagnostic relationship is remembered and monitored.

## What the test asserts

- **Bare (no clamp) — the wrong state:** per-cycle `Δwtd` stays large (flicker present) **and** `cc` vs
  `BDF2-on-V` land on different flickering states (methods disagree → they diagnose the overshoot).
- **Clamped (`-wtm_surface_exfiltration_to_runoff`) — the fix:** the cycle is suppressed
  (`Δwtd → 0`, clean steady state) **and** the schemes reconcile to machine precision.

## If this test fails

It means the **surface-overshoot handling changed** — either the bare flicker was suppressed, or the clamp
stopped reconciling the schemes. That is *not* a solution regression to silence. Investigate, and if the
change is intended (e.g. the clamp becomes the default, or an active-set free-boundary solve replaces the
overshoot), **update this test on purpose** — do not just relax the thresholds to make it green.

See `finding_lakeshore_flicker`, `finding_lakeshore_bounce_activeset`, and `finding_limit_cycle_management`
for the full story and the management options (clamp, under-relaxation, and the active-set cure).
