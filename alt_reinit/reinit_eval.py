"""alt_reinit/reinit_eval.py: metrics-only, in-process test-set evaluation for R2 (alternative track R, alt_reinit/PLAN.md).

alt_reinit/make_reinit_variant.sh copies this file into the FasterGSReinit method; the 'eval' callback that
alt_reinit/patches/fastergs_reinit.py adds to its Trainer.py calls run_eval. Nothing runs unless FGS_EVAL is set:

FGS_EVAL=every=500,from=5000   evaluate after steps 5000, 5500, ..., up to NUM_ITERATIONS
FGS_EVAL=at=12500              evaluate after step 12500 only (R4 confirmation runs: no other intermediate evaluation)
FGS_EVAL=at=12000:12500        several steps, ':'-separated

"After step s" means once s training steps have run (callback iteration s - 1, after the training step). For each test
view it renders the image the way NeRFICG's final test pass does (renderer.render_image in eval mode, clamped), rounds
result and reference to 8 bits as NeRFICG's PNG save does, and computes PSNR, SSIM and LPIPS-VGG with the same torchmetrics
classes as Methods/Base/Renderer.compute_image_metrics, so the value at NUM_ITERATIONS should equal metrics_8bit.txt.

It is outside the track clock: SEGCLOCK times only training_iteration and CBCLOCK only the method's callbacks. It
leaves training untouched: the CPU / CUDA / Python / NumPy RNG states are saved and restored, and the number of
Gaussians and a checksum of the parameters are compared before and after (EVAL ... check=pass|FAIL; FAIL aborts).

Writes 'EVAL step=.. psnr=.. ssim=.. lpips=.. n_images=.. N=.. t_eval=.. check=..' to FGS_LOG and one JSON line per
evaluation, with the per-image values, to <run>/fgs_eval/metrics.jsonl.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalConfig:
    labels: list[int] = field(default_factory=list)   # steps after which to evaluate (1-based step counts)

    @property
    def schedule(self) -> tuple[int, int, int]:
        """(start, end, stride) in NeRFICG callback iterations (0-based): the callback runs at label - 1."""
        its = [s - 1 for s in self.labels]
        if len(its) == 1:
            return its[0], its[0], 1
        stride = 0
        for a, b in zip(its, its[1:]):
            stride = math.gcd(stride, b - a)
        return its[0], its[-1], stride

    def describe(self) -> str:
        if len(self.labels) > 3:
            return f'{len(self.labels)} evaluations, steps {self.labels[0]}..{self.labels[-1]}'
        return 'steps ' + ','.join(map(str, self.labels))


def parse_eval(spec: str, num_iterations: int) -> EvalConfig:
    kv = {}
    for item in spec.split(','):
        if item.strip():
            k, _, v = item.partition('=')
            kv[k.strip()] = v.strip()
    unknown = set(kv) - {'every', 'from', 'at'}
    if unknown:
        raise ValueError(f'FGS_EVAL: unknown key(s) {sorted(unknown)}')
    if ('at' in kv) == ('every' in kv):
        raise ValueError('FGS_EVAL: give either every=N[,from=S] or at=S[:S...]')
    if 'at' in kv:
        if 'from' in kv:
            raise ValueError('FGS_EVAL: from= goes with every=')
        labels = sorted({int(x) for x in kv['at'].split(':')})
    else:
        every = int(kv['every'])
        first = int(kv.get('from', every))
        if every < 1 or first < 1:
            raise ValueError('FGS_EVAL: every and from must be >= 1')
        labels = list(range(first, num_iterations + 1, every))
    if not labels or labels[0] < 1 or labels[-1] > num_iterations:
        raise ValueError(f'FGS_EVAL: steps must lie in 1..{num_iterations}, got {labels[:3]}..')
    return EvalConfig(labels)


@dataclass
class _State:
    metrics: dict | None = None
    gt: list | None = None
    jsonl: Path | None = None


def _quantize(x):
    """NeRFICG's save_image rounding (clamp, *255 + 0.5, truncate) followed by load_images' /255."""
    return x.clamp(0.0, 1.0).mul(255.0).add(0.5).byte()


def _fingerprint(gs) -> tuple:
    import torch
    with torch.no_grad():
        parts = [gs.means, gs.raw_scales, gs.raw_rotations, gs.raw_opacities, gs.sh_coefficients_0, gs.sh_coefficients_rest]
        return int(gs.means.shape[0]), tuple(float(p.detach().double().sum()) for p in parts)


def run_eval(trainer, iteration: int, dataset) -> None:
    import numpy as np
    import torch
    import torchmetrics
    from . import reinit_trainer as rt
    step = iteration + 1
    cfg = trainer._fgs_ecfg
    if step not in cfg.labels:
        return
    st = getattr(trainer, '_fgs_estate', None)
    if st is None:
        import Framework
        from Datasets.utils import apply_background_color  # noqa: F401  (import check before the first evaluation)
        torch.hub.set_dir(str(Framework.Directories.CACHE_DIR))
        dev = Framework.config.GLOBAL.DEFAULT_DEVICE
        st = _State(metrics=dict(
            psnr=torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(dev),
            ssim=torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(dev),
            lpips=torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(dev)))
        st.jsonl = Path(trainer.output_directory) / 'fgs_eval' / 'metrics.jsonl'
        st.jsonl.parent.mkdir(parents=True, exist_ok=True)
        trainer._fgs_estate = st
        rt._log(trainer, f'EVAL_CONFIG {cfg.describe()} schedule={cfg.schedule}')
    from Datasets.utils import apply_background_color
    first = st.gt is None
    gib = 1024 ** 3

    def mem(tag: str) -> None:   # first evaluation only: where the evaluator's own memory goes
        if first:
            rt._log(trainer, f'EVAL_MEM step={step} at={tag} alloc_gib={torch.cuda.memory_allocated() / gib:.2f} '
                             f'peak_gib={torch.cuda.max_memory_allocated() / gib:.2f}')
    mem('start')
    rt._sync()
    t0 = time.perf_counter()
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), random.getstate(), np.random.get_state())
    gs = trainer.model.gaussians
    fp0 = _fingerprint(gs)
    mem('fingerprint')
    was_training = getattr(trainer.model, 'training', True)   # training_iteration calls model.train() every step anyway
    trainer.model.eval()
    dataset.test()
    vals = {k: [] for k in st.metrics}
    try:
        if st.gt is None:   # the references never change: rounded to 8 bits once, kept on the device as uint8
            st.gt = []
            for view in dataset:
                g = view.rgb
                if (a := view.alpha) is not None:
                    g = apply_background_color(g, a, view.camera.background_color)
                st.gt.append(_quantize(g))
            mem(f'references ({len(st.gt)} images, {tuple(st.gt[0].shape)})')
        for i, view in enumerate(dataset):
            out = trainer.renderer.render_image(view, to_chw=True, benchmark=False)['rgb']
            res = _quantize(out).float().div(255.0)[None]
            tgt = st.gt[i].float().div(255.0)[None]
            if i == 0:
                mem('render image 0')
            for k, m in st.metrics.items():
                vals[k].append(float(m(res, tgt)))
                if i == 0:
                    mem(f'metric {k} image 0')
            del out, res, tgt
    finally:
        for m in st.metrics.values():
            m.reset()
        dataset.train()
        if was_training:
            trainer.model.train()
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        random.setstate(rng[2])
        np.random.set_state(rng[3])
    fp1 = _fingerprint(gs)
    mem('end')
    rt._sync()
    dt = time.perf_counter() - t0
    ok = fp0 == fp1
    mean = {k: sum(v) / len(v) for k, v in vals.items()}
    rt._log(trainer, f'EVAL step={step} psnr={mean["psnr"]:.6f} ssim={mean["ssim"]:.6f} lpips={mean["lpips"]:.6f} '
                     f'n_images={len(vals["psnr"])} N={fp1[0]} t_eval={dt:.3f} check={"pass" if ok else "FAIL"}')
    with open(st.jsonl, 'a') as f:
        f.write(json.dumps(dict(step=step, N=fp1[0], t_eval=dt, mean=mean, per_image=vals)) + '\n')
    if not ok:
        raise RuntimeError(f'EVAL at step {step} changed the model: {fp0} -> {fp1}')
    if getattr(trainer, 'requires_empty_cache', False):
        torch.cuda.empty_cache()
