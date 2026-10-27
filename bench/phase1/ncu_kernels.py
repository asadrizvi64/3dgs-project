"""Nsight Compute driver and table for the blend backward kernels.

Run mode (under ncu --profile-from-start off): one forward of a synthetic parity scene, then one _C.backward per
mode inside cudaProfilerStart/Stop, densification with the abs gradient (as in garden training):
    ncu --profile-from-start off -k regex:blend_backward --metrics ... python bench/phase1/ncu_kernels.py run garden_res
Table mode: summarize an `ncu -i <rep> --csv --page raw --metrics ...` export, one row per launch:
    python bench/phase1/ncu_kernels.py table <file.csv> <tag>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

METRICS = [
    'gpu__time_duration.sum', 'launch__grid_size', 'launch__block_size', 'launch__registers_per_thread',
    'sm__warps_active.avg.pct_of_peak_sustained_active', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__inst_executed.sum', 'smsp__thread_inst_executed_per_inst_executed.ratio', 'lts__t_bytes.sum',
    'dram__bytes_read.sum', 'lts__t_requests_op_red.sum', 'lts__t_requests_op_atom.sum',
    'smsp__inst_executed_op_shfl.sum', 'smsp__sass_inst_executed_op_shared_ld.sum',
]
STALLS = ['long_scoreboard', 'short_scoreboard', 'mio_throttle', 'lg_throttle', 'math_pipe_throttle', 'wait',
          'not_selected', 'no_instruction', 'barrier', 'branch_resolving']
METRICS += [f'smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio' for s in STALLS]


def run(scene_name: str, modes: list[int]) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'phase1' / 'tests'))
    from test_parity import SCENES, _C, make_inputs
    scene = next(s for s in SCENES if s.name == scene_name)
    params, settings, extra = make_inputs(scene, 0, False)
    means, scales, rotations, opacities, sh0, sh_rest = params
    image, pb, tb, ib, bb, n_inst, n_buckets, sel = _C.forward(*params, *settings)
    grad_image = torch.randn(image.shape, generator=extra['generator'], device='cuda')
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for mode in modes:
        di = torch.zeros(3, scene.n, device='cuda')
        _C.backward(di, grad_image, image, means, scales, rotations, opacities, sh_rest, pb, tb, ib, bb,
                    *settings, n_inst, n_buckets, sel, mode)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f'profiled scene={scene_name} buckets={n_buckets} modes={modes}')


def table(path: str, tag: str) -> None:
    rows = list(csv.reader(open(path)))
    hdr, units, data = rows[0], rows[1], rows[2:]
    ix = {h: i for i, h in enumerate(hdr)}

    def v(r: list[str], k: str) -> float:
        try:
            return float(r[ix[k]].replace(',', ''))
        except (KeyError, ValueError):
            return float('nan')

    def us(r: list[str]) -> float:
        scale = {'ns': 1e-3, 'nsecond': 1e-3, 'us': 1.0, 'usecond': 1.0, 'ms': 1e3, 'msecond': 1e3}
        return v(r, 'gpu__time_duration.sum') * scale.get(units[ix['gpu__time_duration.sum']], 1.0)

    def mb(r: list[str], k: str) -> float:
        scale = {'byte': 1e-6, 'Kbyte': 1e-3, 'Kbyte': 1e-3, 'Mbyte': 1.0, 'Gbyte': 1e3}
        return v(r, k) * scale.get(units[ix[k]], 1.0)

    def name(r: list[str]) -> str:
        n = r[ix['Kernel Name']]
        if 'macro' in n:
            t = n[n.index('<') + 1:n.index('>')].replace(' ', '')
            return f'macro<{t}>'
        return 'stock(taming)'

    print(f'NCU {tag}: {"kernel":28s} {"us":>7} {"regs":>4} {"occ%":>5} {"sm%":>5} {"inst_M":>7} {"thr/inst":>8} '
          f'{"shfl_M":>6} {"L2_MB":>7} {"red_K":>7} | ' + ' '.join(f'{s[:6]:>6}' for s in STALLS))
    for r in data:
        if 'blend_backward' not in r[ix['Kernel Name']]:
            continue
        stalls = ' '.join(f'{v(r, f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio"):6.2f}' for s in STALLS)
        print(f'NCU {tag}: {name(r):28s} {us(r):7.1f} {v(r, "launch__registers_per_thread"):4.0f} '
              f'{v(r, "sm__warps_active.avg.pct_of_peak_sustained_active"):5.1f} '
              f'{v(r, "sm__throughput.avg.pct_of_peak_sustained_elapsed"):5.1f} {v(r, "smsp__inst_executed.sum") / 1e6:7.2f} '
              f'{v(r, "smsp__thread_inst_executed_per_inst_executed.ratio"):8.1f} '
              f'{v(r, "smsp__inst_executed_op_shfl.sum") / 1e6:6.2f} {mb(r, "lts__t_bytes.sum"):7.1f} '
              f'{(v(r, "lts__t_requests_op_red.sum") + v(r, "lts__t_requests_op_atom.sum")) / 1e3:7.0f} | {stalls}')


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], [int(m) for m in (sys.argv[3] if len(sys.argv) > 3 else '0,1,3,5,7,9,15').split(',')])
    elif sys.argv[1] == 'metrics':
        print(','.join(METRICS))
    else:
        table(sys.argv[2], sys.argv[3])
