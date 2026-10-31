# R1 audit, job 4503153: mask inspection (gate G3) and notes

Job 4503153 (node c95, H100, repo 4607ea4), garden 1/4 and room 1/2, seed 0, 30k stock training with the audit at
k = 2000 and 5000. R0 that preceded it: job 4503144 (JOB OK). Tables and G1/G2: [`audit-4503153-analysis.md`](audit-4503153-analysis.md).
Full audit output (depth/status npz) archived at `/projects/p_scads_mlhpc/syas272h/3dgs-archive/alt_reinit/4503153/`.

All 16 mask images were inspected (`run_4503153/runs/ar1_{garden,room}_audit_s0/reinit_audit/k_0{2,5}000/masks/view*.jpg`).
Colours: green kept, red geometry-rejected, orange depth mixture, blue low alpha, black empty.

## k = 2000: the model has no surfaces yet

Both scenes, all 8 images: almost the whole frame is orange (mixture). The depth-std panel is white (rel. std ≥ 0.2)
nearly everywhere; only the garden table top and vase, and the room's near coffee table / sofa, are dark. Kept pixels
(0.6% garden, 0.5% room) are too sparse to see at this scale.

- **Red (geometry) sits on the nearest, most textured surfaces**: the garden table's slats and the vase
  (`garden/k_02000/view043.jpg`, `view128.jpg`, `view085.jpg`), and paving close to the camera (`view000.jpg`, `view085.jpg`
  bottom right); in room the coffee table top and frame and the sofa cushions (`room/k_02000/view000.jpg`, `view145.jpg`).
  These are the only places with single-layer depth at this point, so they are the only places the geometric test runs.
- **Blue (low alpha)**: garden hedges, brick wall and far background (`view000`, `view043`), and gaps in the lawn and
  paving; room dark carpet and the sofa corner near the camera (`room/k_02000/view072.jpg`, `view217.jpg`, `view145.jpg`).
- **Reading:** at k = 2000 the reinitialization would touch a sliver of the scene. With target `match` it adds 10,886
  rows in garden and 4,270 in room (<1% of N). Garden's stock-15k target (4,166,000) is **unreachable** at k = 2000
  (max 2,081,426 even at stride 1). Room's is reachable only by adding 896k rows drawn from the ~0.5% kept pixels.

## k = 5000: rejections follow edges, dark or glossy surfaces, and large Gaussians

- **Green (kept)**: the textured near and mid-ground. Garden table slats, the vase, paving, the lawn's foreground, the
  plant in `view043`, the metal spheres (`garden/k_05000/view000.jpg`, `view043.jpg`, `view085.jpg`, `view128.jpg`). Room
  armchair and sofa fabric, the yellow book, the textured dark carpet near the camera, plant leaves, and the side-table frame
  (`room/k_05000/view000.jpg`, `view145.jpg`, `view072.jpg`, `view217.jpg`).
- **Red (geometry)**, by type:
  - edges and thin structures: table legs and the table rim (garden), chair and table frames, the speaker stand (room);
  - glass and reflections: the house windows, the metal grate and the garden chair (`garden/k_05000/view043.jpg`,
    `view128.jpg`); the TV screen and glass TV stand (`room/k_05000/view000.jpg`, `view217.jpg`);
  - **whole elliptical regions on the room floor, carpet and sofa** (`room/k_05000/view145.jpg`, `view217.jpg`): red and
    orange patches whose outlines match single large Gaussians in the depth-std panel. This is the effect PLAN.md predicted:
    a large Gaussian gives its centre depth over its whole footprint. The rejection is a property of that Gaussian, not of
    the photo.
- **Orange (mixture)**: the garden table's dark slate centre ring in every view (dark and specular); the mid-ground lawn;
  most of the room's glossy wood floor (`room/k_05000/view072.jpg`, `view217.jpg`); curtains.
- **Blue (low alpha)**: garden hedges, brick wall and far background; foreground paving at the bottom left of
  `garden/k_05000/view128.jpg`, which is near the camera and well textured. That one is unexpected and probably an image-edge
  view with little training coverage. In room: the TV screen, parts of the sofa near the camera (`view145.jpg`) and a band
  along the sofa in `view072.jpg`.

## What this means for the gates

- G3 is done: the masks have been inspected and written up. Nothing in them shows a bookkeeping error: kept pixels sit
  on surfaces that look correct, and rejections follow edges, glass and specular areas, and large-Gaussian footprints.
- The pre-registered rule picks **k = 2000 for both scenes** (earliest passing G1–G3). The masks show that at
  k = 2000 the filter keeps almost nothing. In practice the k = 2000 `reinit-same` arm would be nearly identical to
  `stop`, and garden's `reinit-target` arm cannot be built at k = 2000 at all. **Flagged for the user; the rule was not
  changed.** k = 5000 is where the reinitialization changes a real share of the model (garden removes 1.38M rows, room
  205k).

## Other things noticed (not fixed)

- **The coverage check is saturated and cannot detect holes.** `pixels_covered_before` equals every pixel of the 16
  diagnostic views (garden 17,431,680 ≈ 16 × 1297 × 840; room 25,858,656), so with every opacity forced to 0.99 the
  silhouette already fills the whole frame. Holes / newly covered: garden k = 2000 0 / 0, k = 5000 65 / 10; room 0 / 0 at
  both k. A useful version needs a lower forced opacity, or coverage counted only from the replaced region. Not fixed here.
- **Neighbour selection at k = 2000:** 71 garden views and 46 room views have no neighbours, and the median triangulation
  angle is 88° (garden) / 35° (room), against 8.6° / 14.8° at k = 5000. With so few candidates per view, the
  "largest share in frustum" choice probably picks views across the scene.
- **Agreement with the 30k model at k = 2000** is no better for kept than for geometry-rejected pixels (garden: 59.6% vs
  66.0% within 1%; room: 60.2% vs 63.1%). At k = 5000 kept is clearly better (91.8% vs 87.5%; 86.4% vs 72.5%). This is
  agreement, not accuracy.
- G2 at garden k = 2000 passes on a weak kept class: 18.7% consistent but 45.9% inconsistent (ZNCC < 0.3).
- The R0 job's `^CHECK` grep misses the two-layer GPU depth check, because pytest's progress dot is on the same line.
  It ran and passed (`run_4503144/gpu_tests.log`: D = 1.9995, rel. std 0.500).
