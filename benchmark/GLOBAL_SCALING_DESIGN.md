# Design Note: the path to a global WTM — a memory problem, not a compute problem

**Date:** 2026-07-26
**Branch:** `bdf2-adaptive-dt` (strategy note; not tied to that branch's code)
**Status:** DESIGN / strategic scoping. For deciding direction, not a commitment.
**Authors:** Andy Wickert + Claude
**Companions:** `DISTRIBUTED_ARP_DESIGN.md` (GW-side memory), `PICARD_MG_DESIGN.md`,
`BDF2_ADAPTIVE_DESIGN.md` (GW compute/time-stepping).

## 1. The reframe

Groundwater was the obstacle to run *speed*. With the distributed Picard solve
(`DISTRIBUTED_ARP_DESIGN.md`, `PICARD_MG_DESIGN.md`) and BDF2 + adaptive stepping
(`BDF2_ADAPTIVE_DESIGN.md`), a run that used 3-month / 1-year steps can jump to ~10-year
steps — **~10–40× fewer cycles** — and each cycle's GW solve is distributed and cheap. GW is
no longer the bottleneck; **the runs are computationally feasible.**

So the question for **global WTM (30″)** is no longer "is it fast enough?" but **"does it
fit in memory?"** Any further FSM *compute* win matters little next to the GW speedup already
in hand. **The reason to parallelize the surface-water machinery is memory, not speed** —
compute relief is a hoped-for bonus, not the motivation.

**Notation.** `N` = the total number of grid cells over the global domain (the DMDA size). At
**30″ global this is 933,120,000** (43200 × 21600 ≈ **9.3×10⁸**) — a maximum; the poles (and
ocean) may be excluded. `P` = MPI ranks. "Full grid" means an O(N) array held whole on one
rank — so a single `double` full-grid field at 30″ is ~7.5 GB, and the DH build needs several.

The GW step is capped at the **FSM cadence** (~10 yr: surface water must be routed before it
accumulates unphysically), so the realized picture is *one cheap, distributed GW step per FSM
call*. That is fine for feasibility; it just means the surface-water side is what's left to
make global.

## 2. The global memory ledger — what still holds the full grid?

| structure | global memory today | size | status |
|---|---|---|---|
| GW solve arrays (T, S, head, RHS, …) | distributed on the DMDA | O(N/P)/rank | ✅ done (the flip) |
| rank-0 `arp` (GW-side full-grid fields) | full grid on rank 0 | O(N) | ⏳ `DISTRIBUTED_ARP_DESIGN.md` (Class A/B/C) |
| **DH *build*: DEM + per-cell labels + flowdirs** | **full grid on rank 0** | **several O(N)** | ❌ **the last, and largest, global-memory ceiling** |
| DH *tree* (depressions: volumes, spill pts, links) | on rank 0 | O(#depressions) ≪ N | rides along with the build |
| FSM *traversal* (moves water through the tree) | reads `wtd` + tree on rank 0 | `wtd` is O(N) but already distributed on the DMDA | light once the build is distributed |

At 141M on a fat node the full grid fits (tens of GB); at **global 30″ it does not**. GW
memory is being distributed. **The remaining — and dominant — barrier is the DepressionHierarchy
*build*:** the priority-flood over the DEM produces *several* full-grid O(N) arrays (the DEM
itself, the per-cell depression **labels**, flowdirs), and that is far more memory-intensive
than the FSM *traversal*, which works on the depression **tree** (≪ N) plus `wtd` (already
distributed). So the crux is not the graph traversal but **building the hierarchy from the
global DEM**: *global WTM ⟺ a distributed DH build.* FSM traversal on the resulting distributed
hierarchy is comparatively light.

Helpfully, **DH is rebuilt only decadal–centennial** (topography changes slowly — GIA/isostasy),
*rarer* than FSM runs (~10 yr cadence). So the distributed build is an **occasional, amortized**
cost, not a per-cycle one — which further shifts the emphasis onto making the build *fit* rather
than making it *fast*.

## 3. Distributing the DH build (memory-first) — Barnes' tiled priority-flood

The memory-critical target is the **DH build from the global DEM**, and it is *exactly* what
**Barnes (2016), parallel priority-flood** was designed for (*Computers & Geosciences*,
"trillion-cell DEMs on desktops or clusters"). The FSM algorithm itself is Barnes, Callaghan &
Wickert — you are a co-author, so this is a collaboration, not a reinvention.

**Priority-flood *is* the engine of the DH build.** The DH is built by a priority-flood that,
as the flood rises from the domain edges, additionally **records merge events**: each saddle
where two sub-depressions meet becomes a parent node, assembling the depression tree bottom-up.
So *DH-build = priority-flood + merge-tree recording*, and **parallel DH-build = parallel
priority-flood, generalized to carry that tree**:

- the **local** work is Barnes (2016) essentially unchanged — tile-local flood, now also
  recording each tile's local tree — and it distributes exactly the O(N) arrays (DEM, labels)
  that make the flood scale to trillion cells;
- the **generalization** is the global boundary step: parallel priority-flood reconciles
  cross-tile fill *levels*; parallel DH must additionally **stitch the cross-tile depression
  *tree*** (unify tile-spanning depressions, record boundary saddle/merge events into one global
  hierarchy). The boundary graph the flood already builds does **double duty** — its cross-tile
  spill structure is most of what the hierarchy's cross-tile links need. **This tree-stitch is
  the genuinely novel piece** (a distributed DH is unpublished; the flood is 2016, DH is 2020).

Concretely, the memory-distributing shape:

- **Each rank owns its DMDA subdomain's DEM** (co-locate with the GW decomposition, so `topo`
  and `wtd` are *already there* — no gather) and runs the priority-flood **locally**, producing
  its tile's per-cell labels and local depression hierarchy. **This is where the O(N) memory
  is distributed:** the DEM and label arrays become O(N/P) per rank.
- **Inter-tile depressions are stitched via a global *boundary graph*** — a structure over only
  the tile-edge spill points, size ~O(boundary) **≪ N**. This small object is the *only*
  whole-domain data any rank holds, and it carries the cross-tile parent/child links of the
  hierarchy. (Barnes' 2016 join step.)
- **FSM traversal** then runs on the distributed hierarchy: tile-local water movement, with
  cross-tile spill/merge exchanged through the boundary graph until the global fill/spill state
  is consistent. Because the tree and `wtd` are light/already-distributed, this rides on the
  distributed build almost for free — the hard, memory-defining work was the build.

**Memory outcome (the point):** per-rank footprint → **O(N/P) + O(boundary)** → no rank holds a
full-grid array → **global becomes feasible.** The build parallelizes too (and FSM with it), but
that compute relief is the bonus, not the goal.

## 4. Sequencing to global

1. **GW solve distributed** — ✅ done (the flip).
2. **GW steps big + cheap** (BDF2 / adaptive) — ✅ done; makes runs computationally feasible.
3. **Distribute the rank-0 GW `arp`** — `DISTRIBUTED_ARP_DESIGN.md` (Class A/B/C). Removes the
   GW-side memory ceiling.
4. **Distribute the DH build** (§3) — removes the *last, and largest,* global-memory ceiling
   (the full-grid DEM + labels of the priority-flood). FSM traversal follows on the distributed
   hierarchy. **This is the enabling step for global 30″.**

Compute-side surface-water work (e.g. OpenMP FSM for the single-node 141M case) is **orthogonal
and optional** — a speed bonus for the non-global regime, not on the critical path to global.

## 5. Risks / open questions (memory-focused)

- **Boundary-graph size — the #1 thing to de-risk (but partly reassured).** The whole memory
  claim rests on the inter-tile structure being ≪ N. **Barnes 2016 already floods trillion-cell
  DEMs**, which is empirical evidence that the flood's boundary structure *is* ≪ N in practice;
  the DH stitch adds only sparse tree links (~O(#cross-tile depressions)) on top. Still worth
  bounding against real global DEM tiles for **pathological cases** (vast flats spanning many
  tiles) before committing.
- **Cross-tile hierarchy stitching (in the build).** Correctly merging tile-local depression
  labels/links into one global hierarchy across rank boundaries is the algorithmic core of the
  distributed *build* — this is where the research effort concentrates (Barnes' 2016 join).
- **Cross-tile water conservation (in the traversal).** FSM spilling/merging water across rank
  boundaries, exactly and iteratively, is real but **secondary** — the tree and `wtd` are light,
  so this is the smaller half of the problem.
- **DH rebuild cadence — low risk (resolved).** Rebuilds are decadal–centennial (GIA/isostasy),
  *rarer* than FSM, so the distributed build is an **amortized occasional** cost. Emphasis is on
  making the build *fit*, not making it *fast*.
- **Load balance — deprioritized.** Depressions cluster spatially, but FSM traversal is fast, so
  traversal imbalance barely matters. What matters is the **priority-flood build's** balance,
  which Barnes' tile scheme already targets — so keep the DMDA (GW-co-located) decomposition and
  don't chase a depression-aware partition.

## 6. Bottom line / decision framing

Global WTM is a **memory** goal, and after the GW `arp` is distributed, the **DH build — the
priority-flood's full-grid DEM + labels — is the single largest remaining full-grid structure**
(the FSM traversal is comparatively light, and DH rebuilds are rare/amortized). So distributing
the **build** (Barnes' tiled priority-flood + boundary-graph stitching) is *the* enabling step,
justified by memory alone with compute as a bonus. It is a **research-grade, multi-month effort**
best **co-scoped with Richard Barnes** (RichDEM / parallel priority-flood author and FSM
co-author). Recommended precursor: **bound the boundary-graph size against real global DEM
tiles** — that de-risks the one assumption the whole plan rests on, before the full build.
