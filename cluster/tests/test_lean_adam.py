"""GPU checks for the lean Adam kernel (adam_step_sparse) and the n_touched_tiles visibility export.

    python -I cluster/tests/test_lean_adam.py      (inside the fastergs venv, on a GPU node)

1. sparse kernel, fp32 moments, no mask   == original adam_step (bit-exact)
2. with a mask: visible rows == original, invisible rows untouched (params and moments)
3. bf16 moments + stochastic rounding: 3000 steps vs fp32 reference stay close; the beta2=0.999 EMA does
   not stall (round-to-nearest bf16 would), checked against a deterministic-rounding control
4. n_touched_tiles from a real forward pass: every primitive with a nonzero gradient is marked visible
"""
import math
import sys

import torch
from FasterGSCudaBackend import _C

torch.manual_seed(0)
dev = 'cuda'
ok = True


def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f'{"PASS" if cond else "FAIL"}  {name}  {detail}', flush=True)


def adam_ref(p, g, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-15):
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    denom = v.sqrt() / math.sqrt(1 - b2 ** step) + eps
    p.addcdiv_(m, denom, value=-lr / (1 - b1 ** step))


N, D = 100_003, 45
empty = torch.empty(0, dtype=torch.int32, device=dev)

# 1. no mask, fp32 moments: identical to the original kernel
p0 = torch.randn(N, 15, 3, device=dev)
pa, pb = p0.clone(), p0.clone()
ma, va, mb, vb = (torch.zeros_like(p0) for _ in range(4))
for step in range(1, 51):
    g = torch.randn_like(p0) * 1e-3
    _C.adam_step(g, pa, ma, va, step, 1e-3, 0.9, 0.999, 1e-15)
    _C.adam_step_sparse(g, pb, mb, vb, empty, step, 1e-3, 0.9, 0.999, 1e-15)
check('1 dense sparse-kernel == original', torch.equal(pa, pb) and torch.equal(ma, mb) and torch.equal(va, vb),
      f'max|dp|={float((pa - pb).abs().max()):.3e}')

# 2. masked update
vis = (torch.rand(N, device=dev) < 0.4).int() * torch.randint(1, 50, (N,), device=dev, dtype=torch.int32)
pa, pb = p0.clone(), p0.clone()
ma, va = torch.randn_like(p0).abs(), torch.rand_like(p0)
mb, vb = ma.clone(), va.clone()
m_before, v_before = ma.clone(), va.clone()
g = torch.randn_like(p0) * 1e-3
_C.adam_step(g, pa, ma, va, 7, 1e-3, 0.9, 0.999, 1e-15)
_C.adam_step_sparse(g, pb, mb, vb, vis, 7, 1e-3, 0.9, 0.999, 1e-15)
sel = vis > 0
check('2a visible rows == original', torch.equal(pa[sel], pb[sel]) and torch.equal(ma[sel], mb[sel]) and torch.equal(va[sel], vb[sel]),
      f'{int(sel.sum())}/{N} visible')
check('2b invisible rows untouched', torch.equal(pb[~sel], p0[~sel]) and torch.equal(mb[~sel], m_before[~sel]) and torch.equal(vb[~sel], v_before[~sel]))
for shape in [(N, 3), (N, 1), (N, 4), (N, 1, 3)]:   # other group shapes (features per primitive 3, 1, 4, 3)
    q = torch.randn(*shape, device=dev); q0 = q.clone()
    mq, vq = torch.zeros_like(q), torch.zeros_like(q)
    _C.adam_step_sparse(torch.randn_like(q), q, mq, vq, vis, 1, 1e-2, 0.9, 0.999, 1e-15)
    check(f'2c mask indexing for shape {tuple(shape)}', torch.equal(q[~sel], q0[~sel]) and bool((q[sel] != q0[sel]).all()))

# 3. bf16 moments with stochastic rounding vs fp32 reference
pr = torch.randn(N, 15, 3, device=dev)
pinit = pr.clone()
pl = pr.clone()
mr, vr = torch.zeros_like(pr), torch.zeros_like(pr)
ml, vl = torch.zeros_like(pr, dtype=torch.bfloat16), torch.zeros_like(pr, dtype=torch.bfloat16)
scale = torch.rand(N, 1, 1, device=dev) * 1e-3 + 1e-6          # per-Gaussian gradient scale, wide range
bias = torch.randn_like(pr) * scale
lr = 1.25e-4
for step in range(1, 3001):
    g = bias + torch.randn_like(pr) * scale
    _C.adam_step(g, pr, mr, vr, step, lr, 0.9, 0.999, 1e-15)
    _C.adam_step_sparse(g, pl, ml, vl, empty, step, lr, 0.9, 0.999, 1e-15)
rel_v = ((vl.float() - vr).abs() / (vr.abs() + 1e-30)).median()
rel_m = (ml.float() - mr).norm() / mr.norm()
dr, dl = pr - pinit, pl - pinit
rel_upd = (dr - dl).norm() / dr.norm()
check('3a bf16 v tracks fp32 v (median rel err < 2%)', float(rel_v) < 0.02, f'median rel err {float(rel_v):.4f}')
check('3b bf16 m tracks fp32 m (rel norm err < 5%)', float(rel_m) < 0.05, f'rel norm err {float(rel_m):.4f}')
check('3c total parameter update agrees (rel err < 2%)', float(rel_upd) < 0.02, f'rel err of 3000-step update {float(rel_upd):.4f}')

# stall control: constant gradient after a warm start; v must approach g^2 from below with SR,
# while round-to-nearest bf16 storage gets stuck once 0.001 * (g^2 - v) < half an ulp of v
g_const = torch.full((1 << 20,), 1.0, device=dev)
v_sr = torch.full_like(g_const, 0.5).to(torch.bfloat16)
m_sr = torch.zeros_like(v_sr)
q = torch.zeros_like(g_const)
v_rn = torch.full_like(g_const, 0.5)
for step in range(1, 3001):
    _C.adam_step_sparse(g_const, q, m_sr, v_sr, empty, 10_000 + step, 0.0, 0.9, 0.999, 1e-15)
    v_rn = (0.999 * v_rn + 0.001 * g_const * g_const).to(torch.bfloat16).float()
expected = 1.0 - 0.5 * 0.999 ** 3000
check('3d SR EMA converges (no stall)', abs(float(v_sr.float().mean()) - expected) < 0.005,
      f'SR mean {float(v_sr.float().mean()):.4f}, exact {expected:.4f}, round-to-nearest {float(v_rn.mean()):.4f}')

# 4. visibility export from a real forward pass
from FasterGSCudaBackend.torch_bindings.rasterization import _Rasterize, RasterizerSettings, diff_rasterize
n = 20_000
sign = torch.where(torch.rand(n, 1, device=dev) < 0.5, -1.0, 1.0)   # half in front, half behind (either z convention)
means = (torch.randn(n, 3, device=dev) * torch.tensor([2.0, 2.0, 1.0], device=dev) + sign * torch.tensor([0, 0, 5.0], device=dev)).requires_grad_()
scales = torch.full((n, 3), math.log(0.03), device=dev).requires_grad_()
rot = torch.zeros(n, 4, device=dev); rot[:, 0] = 1; rot.requires_grad_()
opac = torch.zeros(n, 1, device=dev).requires_grad_()
sh0 = torch.rand(n, 1, 3, device=dev).requires_grad_()
shr = torch.zeros(n, 15, 3, device=dev).requires_grad_()
w2c = torch.eye(4, device=dev)
settings = RasterizerSettings(w2c=w2c.contiguous(), cam_position=torch.zeros(3, device=dev), bg_color=torch.zeros(3, device=dev),
                              active_sh_bases=1, width=320, height=240, focal_x=300.0, focal_y=300.0, center_x=160.0, center_y=120.0,
                              near_plane=0.2, far_plane=1000.0, proper_antialiasing=False)
_Rasterize.track_visibility = True
img = diff_rasterize(means, scales, rot, opac, sh0, shr, torch.empty(0), settings)
img.sum().backward()
vmask = _Rasterize.last_visibility
gnz = means.grad.abs().sum(1) > 0
check('4a visibility shape/dtype', vmask is not None and vmask.shape == (n,) and vmask.dtype == torch.int32)
check('4b every primitive with a gradient is visible', bool((vmask[gnz] > 0).all()),
      f'visible {int((vmask > 0).sum())}/{n}, with grad {int(gnz.sum())}')
check('4d some primitives are visible', int((vmask > 0).sum()) > 0)
check('4c some primitives are culled', int((vmask == 0).sum()) > 0, f'culled {int((vmask == 0).sum())}')

print('ALL PASS' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
