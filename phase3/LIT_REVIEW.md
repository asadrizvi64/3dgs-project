# Literature and novelty audit for Experiments 1R and 1A (2026-11-12)

**Purpose.** To decide whether the results of Experiments 1R ([EXP1R_RESULTS.md](EXP1R_RESULTS.md)) and 1A
([EXP1A_RESULTS.md](EXP1A_RESULTS.md)) are a new scientific contribution, and what would be needed to make one. No GPU
experiments were run for this audit. Experiment 2 stays frozen. The cluster checkout was not touched.

## How the audit was done, and its limits

- **Keyword searches** (about 15) on:
  - 3DGS-MCMC, its ablations, and follow-ups;
  - mid-training or hybrid densification;
  - time-to-quality and convergence measurement;
  - densification schedules and stopping rules;
  - opacity resets;
  - adaptive or learned density control.
- **Citation scan:** every paper citing 3DGS-MCMC in Semantic Scholar's index (the full list, read in six chunks),
  filtered by title and abstract for:
  - switching or combining ADC with MCMC;
  - time-to-quality measurement;
  - analysis of MCMC components or opacity resets;
  - adaptive budget or stopping control.
- **Read in part** (full text fetched and queried):
  - 3DGS-MCMC (NeurIPS 2024): its ablations and timing;
  - Faster-GS (CVPR 2026), including appendix C.3 on its fast MCMC;
  - Decomposing Densification (arXiv 2507.20239);
  - SAD-GS (SIGGRAPH 2026).

  Everything else was read from abstracts or summaries only.
- **Limits:**
  - A citation scan misses work that does not cite 3DGS-MCMC.
  - Summaries can be wrong. Numbers marked "summary" were not checked against the paper.
  - Several papers appeared after mid-2026 and were seen only through search.
  - A "not found" below means not found by this search, not proven absent.
  - Before any submission, a second pass is needed through the 2025–2026 proceedings of CVPR, ICCV, ECCV, SIGGRAPH (and
    Asia), NeurIPS, ICLR and 3DV.

## What the literature already establishes

**3DGS-MCMC** ([Kheradmand et al., NeurIPS 2024][mcmc]):
- **Its three components:** relocation (in place of clone, split and prune), position noise (SGLD), and opacity and
  scale regularisation.
- **Ablations, all for quality, with random initialisation:**
  - without noise, Mip-NeRF 360 PSNR drops from 29.72 to 27.41;
  - without the regulariser, from 29.72 to 23.90;
  - the noise schedule and covariance term are also ablated on Tanks & Temples.
- **No quantitative ablation of relocation** against clone and split: only a 1-D illustration.
- **Time:**
  - one table on *room*: 3DGS reaches 31.7 dB in 25 min; MCMC with a 300k cap reaches 31.8 dB in 21 min;
  - the authors say the method "trains faster when a similar PSNR is desired";
  - no PSNR-over-time curves, no repeated seeds, no time-to-target protocol.
- **Schedule:** MCMC is always used from the start, with a 500-step warm-up, then growth of 5% per step up to a cap. It
  is never switched on mid-training, and never started from a partly trained model.

**Faster-GS** ([Hahlbohm et al., CVPR 2026][fastergs]), the codebase we build on:
- Appendix C.3 fuses the noise injection into one kernel, which they found to be the main bottleneck. "Faster-GS MCMC"
  reaches 28.00 dB in 6 min 17 s on Mip-NeRF 360, against 27.83 dB in 27 min 22 s for the original MCMC code.
- No comparison with Faster-GS's own default ADC. No time-to-quality measure. No mid-training switch.
- The opacity learning rate is halved to 0.025 by default, with a note to use 0.05 with MCMC.

**Fast-training work** generally reports total time at a fixed number of iterations. Some add quality-against-time
curves:
- SAD-GS ([Lyu et al., SIGGRAPH 2026][sad]): curves against FastGS variants, plus a comparison at matched Gaussian count.
- Decomposing Densification ([Huang et al., 2025][decomp]): split and clone phases, 2× faster, final-time tables only.
- DashGaussian, FastGS, Turbo-GS, Mini-Splatting2, EDGS, Improved ADC ([Grubert et al., VISAPP 2025][iadc]).
- gsplat's library paper ([Ye et al., JMLR][gsplat]) times MCMC against default ADC, but at different Gaussian counts
  (1.00 M against 3.24 M).

**This search found no standard time-to-matched-quality protocol with paired seeds.**

**Opacity resets** are criticised qualitatively, and replaced or softened, in:
- Revising Densification ([Rota Bulò et al.][revis]);
- Unified Loss-Aware Density Control;
- Smart Target Point Control ([2605.16158][stpc]);
- Object Learning and Robust 3D Reconstruction.

None of these measures the reset's effect on time to matched quality.

**Adaptive or learned control of the population** is an active area:
- Turbo-GS ([Lu et al., CVPR 2026][turbo]): a convergence-aware budget, setting the growth rate from a power-law fit to
  the loss.
- GS² ([CVPR 2026][gs2]): an ELBO-based stop for densification.
- Smart Target Point Control: drives N to a target count.
- SPARE-GS: adaptive termination under a budget.
- LeGS, "Beyond Heuristics" ([ICML 2026][legs]): a reinforcement-learning policy over clone, split, prune and keep.
- DP-Splat: Dirichlet-process complexity control.
- Taming 3DGS ([Mallick et al., 2024][taming]): a budgeted growth schedule.

**Analyses of MCMC itself:**
- Critiques of the position-only noise: Denoising-GS, Opt3DGS.
- How parameter inheritance in relocation matters: FreeTimeGS++.
- An optimiser analysis run under both frameworks: "A Step to Decouple Optimization in 3DGS" ([ICLR 2026][decouple]).

None measures what each component contributes to convergence speed.

## Literature matrix

Status, judged against what this search found:
- **known**: established in the literature;
- **reproduced**: known, and our experiments confirm it;
- **strengthened**: claimed before on weak evidence, and our evidence is much stronger;
- **apparently new**: not found;
- **partly known**: neighbouring work exists.

| # | question | literature | our evidence | status |
|---|---|---|---|---|
| 1 | MCMC (relocation + noise + regularisation) gives higher final quality than ADC | 3DGS-MCMC; Faster-GS C.3; gsplat; many follow-ups | +0.19 to +0.44 dB, lower LPIPS, about 27% fewer Gaussians (1R, 1A; 2 scenes) | **reproduced** |
| 2 | MCMC reaches a given quality sooner than ADC | 3DGS-MCMC: one scene, one run, wall-clock minutes, its own statement; gsplat at unequal counts | 12 of 12 paired comparisons, 29–42% less training-step time to `ctl`'s final PSNR − 0.05 dB, robust to the target, and callback time measured (under 1%) | **strengthened**: the claim is known; rigorous evidence for it is not |
| 3 | Switching from ADC growth to fixed-population MCMC mid-training | not found. MCMC is always used from the start; neighbours: phased densification (Decomposing), budgeted growth (Taming), early densification (SAD-GS) | the switch at step 5,000 works and replicates | **apparently new**, as a configuration. Conceptually close to MCMC's own grow-then-relocate schedule, so incremental |
| 4 | Noise injection matters | 3DGS-MCMC: for quality, random initialisation | needed for time to matched quality: in 3 of 6 pairs the target is never reached | **reproduced** (quality); the speed framing is new but follows from the quality result |
| 5 | Regularisation matters | 3DGS-MCMC: for quality, random initialisation | needed, but sensitive to the target choice | **reproduced** (quality); speed framing new and weak |
| 6 | Relocation matters, quantitatively | **not found**: 3DGS-MCMC has no quantitative relocation ablation | needed: the arm ends 0.34 / 0.73 dB lower and never reaches the target in 4 of 6 pairs | **apparently new** as a measurement. Small |
| 7 | Opacity learning rate 0.05 is needed with MCMC | used by convention (MCMC code; Faster-GS's note) | not needed for speed; small LPIPS cost on garden, at the limit | **apparently new**, minor |
| 8 | ADC is slow because of opacity resets | resets criticised qualitatively | no: removing them does not speed `ctl` | **apparently new**, a small negative result |
| 9 | Advantage comes from population size or from the policy | not separated in the literature found | 1R's oracle, MCMC at `ctl`'s final count, ends higher still; 1A shows the gain mostly lets MCMC reach the quality at all | **partly new**, a hint only (oracle untested in 1A) |
| 10 | In-training evaluation silently changes training (shared RNG: the LPIPS construction and the view sampler) | not found as a reported pitfall | found, fixed and verified (Experiment 1, v7) | **apparently new**, methodological; may be NeRFICG-specific |
| 11 | Same-seed GPU non-determinism: final PSNR SD 0.05–0.07 dB; single views up to −8 dB on identical runs | not found reported | measured (noise pilot, `mrep`) | **apparently new**, methodological; bears on how 0.1-dB claims are read |
| 12 | Adaptive control of growth and budget | crowded: Turbo-GS, GS², STPC, SPARE-GS, LeGS, DP-Splat, Taming | none | **known area**. A growth-to-relocation switch controller specifically was not found |

## Judgments

### 1. Is the existing work enough for a diagnostic or comparative paper?

**Not yet.**
- **The headline is known.** MCMC is better (row 1), and is said to be faster (row 2). The component ablations for
  quality are in the original paper.
- **What we add is real, but narrow:**
  - rigorous, paired, time-to-matched-quality evidence (row 2);
  - the mid-training switch (row 3);
  - component-for-speed and relocation measurements (rows 6–8);
  - two methodological findings (rows 10, 11).
- **All on 2 scenes**, one codebase and one GPU, with no MCMC-from-scratch comparison.

As it stands this is a strong internal study and a workshop-level note, not a conference paper.

### 2. Is there a genuinely unexplored question?

Three candidates, from strongest to weakest:

1. **Where does MCMC's time-to-quality advantage come from, measured properly?**
   - It could be fewer Gaussians (cheaper steps), the relocation-and-noise policy (better use of the same capacity), or
     simply a higher final quality.
   - It could also depend on *when* the population stops growing.
   - The literature reports MCMC as better, and asserts it is faster, but has not separated these.
   - Our oracle arm (same count as `ctl`, better quality) already suggests the policy matters more than the count.
   - This question is testable with the tools already built: count-matched arms, a switch-time sweep, and both clocks.
2. **Benchmarking methodology for "faster 3DGS" claims.**
   - A paired time-to-matched-quality protocol.
   - The measured same-seed noise floor (0.05–0.07 dB, with per-view failures on identical runs).
   - The evaluation side-effect on training.
   - Many fast-training papers compare at fixed iterations, or across different counts and hardware. A protocol paper,
     or a section, with these measurements would be useful and is not found.
   - Risk: methodological contributions are judged on breadth. This needs several methods and datasets.
3. **The ADC-to-MCMC switch as a method.** Apparently new as a configuration, but conceptually close to MCMC's own
   grow-then-relocate schedule. Alone it is unlikely to be enough.

### 3. Which experiments would that need?

These are proposals only. Each needs its own pre-registration and a go-ahead.

**E1, generalisation and the source of the advantage.** Fresh seeds; all 9 Mip-NeRF 360 scenes; ideally 2 scenes
from Tanks & Temples or Deep Blending.
- **Arms:**
  - `ctl` (ADC);
  - MCMC switched on at 5,000 (as in 1R and 1A);
  - MCMC from step 0 with the cap at `ctl`'s step-5,000 count;
  - MCMC from step 0 with the cap at `ctl`'s final count (the oracle-like, count-matched arm);
  - optionally, the simplified policy without the opacity learning-rate change.
- **Measures:** time to matched quality on both clocks, final quality, per-step cost.
- **Cost:** roughly 27 scene-seeds × about 5 runs × 15–25 min, or about 40–55 GPU-hours, mostly evaluation.
- **Cheaper alternative:** start with 3–4 scenes chosen in advance, to check the design.

**E2, switch-time sweep.** MCMC switched on at about 2k, 5k, 10k and 15k on a subset of scenes.
- This is what shows whether the time of the switch matters, and whether the best time differs by scene.
- It is the evidence an adaptive controller would need (question 4).

**Before either:** the second literature pass (above), and a check that the result is not specific to NeRFICG or
Faster-GS. One scene in gsplat's MCMC and default modes would show whether row 2 holds outside our codebase.

### 4. Is an adaptive population controller justified?

**Not now.**
- The area is crowded: Turbo-GS already adapts the growth budget to convergence, GS² stops densification adaptively,
  and LeGS learns density control.
- A controller for when growth should hand over to relocation is not found. But it is only justified if E2 shows that
  the best switch time matters materially *and* differs between scenes.
- If a fixed early switch, or MCMC from step 0, does about as well everywhere, there is nothing for a controller to
  decide.
- Any controller would also have to beat those simple baselines and Turbo-GS-style budget control on the same protocol.

## How to describe the current work, until then

- Experiments 1R and 1A show that a combined MCMC policy, switched on at step 5,000 of Faster-GS training, reaches the
  control's matched quality sooner on two scenes (12 of 12 paired comparisons, training-step clock).
- Leave-one-out evidence shows that relocation, noise injection and regularisation are each needed for this, given the
  rest of the policy. Mostly, without them the policy does not reach the matched quality at all.
- Relocation, noise and regularisation are **not** new (3DGS-MCMC). That MCMC improves quality is **not** new.
- Branching does **not** give final quality identical to uninterrupted training. Check 1 established implementation
  fidelity only.

## Sources

- [3D Gaussian Splatting as Markov Chain Monte Carlo (NeurIPS 2024)](https://arxiv.org/html/2404.09591v3)
- [Faster-GS: Analyzing and Improving Gaussian Splatting Optimization (CVPR 2026)](https://arxiv.org/html/2602.09999)
- [Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification (SIGGRAPH 2026)](https://arxiv.org/html/2604.28016v1)
- [Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction](https://arxiv.org/html/2507.20239)
- [Improving Adaptive Density Control for 3D Gaussian Splatting (VISAPP 2025)](https://arxiv.org/abs/2503.14274)
- [gsplat: An Open-Source Library for Gaussian Splatting](https://arxiv.org/pdf/2409.06765)
- [Revising Densification in Gaussian Splatting](https://arxiv.org/pdf/2404.06109)
- [Smart Target Point Control for Gaussian Splatting Methods](https://arxiv.org/html/2605.16158)
- [Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields](https://arxiv.org/abs/2412.13547)
- [GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting (CVPR 2026)](https://arxiv.org/html/2604.01884)
- [Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting (ICML 2026)](https://proceedings.mlr.press/v306/ning26e.html)
- [Taming 3DGS: High-Quality Radiance Fields with Limited Resources](https://arxiv.org/html/2406.15643v1)
- [A Step to Decouple Optimization in 3DGS (ICLR 2026)](https://arxiv.org/html/2601.16736)

[mcmc]: https://arxiv.org/html/2404.09591v3 "3D Gaussian Splatting as Markov Chain Monte Carlo (NeurIPS 2024)"
[fastergs]: https://arxiv.org/html/2602.09999 "Faster-GS: Analyzing and Improving Gaussian Splatting Optimization (CVPR 2026)"
[sad]: https://arxiv.org/html/2604.28016v1 "Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification (SIGGRAPH 2026)"
[decomp]: https://arxiv.org/html/2507.20239 "Decomposing Densification in Gaussian Splatting for Faster 3D Scene Reconstruction"
[iadc]: https://arxiv.org/abs/2503.14274 "Improving Adaptive Density Control for 3D Gaussian Splatting (VISAPP 2025)"
[gsplat]: https://arxiv.org/pdf/2409.06765 "gsplat: An Open-Source Library for Gaussian Splatting"
[revis]: https://arxiv.org/pdf/2404.06109 "Revising Densification in Gaussian Splatting"
[stpc]: https://arxiv.org/html/2605.16158 "Smart Target Point Control for Gaussian Splatting Methods"
[turbo]: https://arxiv.org/abs/2412.13547 "Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields"
[gs2]: https://arxiv.org/html/2604.01884 "GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting"
[legs]: https://proceedings.mlr.press/v306/ning26e.html "Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting (ICML 2026)"
[taming]: https://arxiv.org/html/2406.15643v1 "Taming 3DGS: High-Quality Radiance Fields with Limited Resources"
[decouple]: https://arxiv.org/html/2601.16736 "A Step to Decouple Optimization in 3DGS (ICLR 2026)"

Also consulted, from abstracts or summaries only:
- Efficient Confidence-Aware Densification (arXiv 2610.13541);
- Metropolis-Hastings Sampling for 3D Gaussian Reconstruction (arXiv 2506.12945);
- AdpSplit (arXiv 2605.06876);
- Denoising-GS (arXiv 2605.14880);
- Opt3DGS (arXiv 2511.13571);
- FreeTimeGS++ (arXiv 2605.03337);
- Efficient multi-view training for 3DGS (arXiv 2506.12727);
- SPARE-GS (arXiv 2607.16624);
- EDGS (arXiv 2504.13204);
- FastGS and DashGaussian, via the papers above;
- the Semantic Scholar list of papers citing 3DGS-MCMC:
  https://api.semanticscholar.org/graph/v1/paper/arXiv:2404.09591/citations
