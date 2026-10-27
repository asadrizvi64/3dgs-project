"""Mock build for the Phase 0 kernels (no GPU, no nvcc needed).

Step 1: every .cu is parsed by g++ as C++17 with -DSPLAT2D_MOCK_CUDA and cuda_shim.h
        force-included, under -Wall -Wextra -Werror. This proves syntax and types only:
        there are no real warps, no shared-memory or memory-model semantics, and nothing is
        executed. (The torch-free CPU reference gets the same g++ warning check.)
Step 2: the CPU extension (splat2d_cpu.cpp + bindings.cpp) is built with
        torch.utils.cpp_extension.load and extra_cflags ["-O2", "-Wall", "-Werror"]
        (on MSVC the equivalent /O2 /W4 /WX, see splat2d.ops.host_cflags) and smoke-run.
Step 3: if nvcc is on PATH, the .cu files are compiled for real with -DWITH_CUDA
        -gencode arch=compute_80,code=sm_80 and the CUDA extension is built. Otherwise this
        prints "nvcc absent: GPU compile deferred to cluster" and exits 0.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from splat2d.ops import CPU_SOURCES, CUDA_SOURCES, host_cflags, load_extension  # noqa: E402

SHIM = "csrc/macrotile/cuda_shim.h"
GXX_WARN = ["-Wall", "-Wextra", "-Werror"]
MOCK_FLAGS = ["-DSPLAT2D_MOCK_CUDA", "-include", SHIM]
SM80 = ["-gencode", "arch=compute_80,code=sm_80"]


def run(cmd: list[str]) -> None:
    # Put the compiler's own directory first on PATH so it loads its own runtime DLLs
    # (on Windows another toolchain earlier on PATH can otherwise crash cc1plus silently).
    env = dict(os.environ)
    env["PATH"] = str(Path(cmd[0]).parent) + os.pathsep + env.get("PATH", "")
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def step1_mock_syntax() -> None:
    gxx = shutil.which("g++")
    if gxx is None:
        raise SystemExit("step 1: g++ not found on PATH")
    for cu in CUDA_SOURCES:
        rel = cu.relative_to(ROOT).as_posix()
        run([gxx, "-std=c++17", "-fsyntax-only", *GXX_WARN, "-x", "c++", *MOCK_FLAGS, rel])
    cpu = CPU_SOURCES[0].relative_to(ROOT).as_posix()
    run([gxx, "-std=c++17", "-fsyntax-only", *GXX_WARN, cpu])
    print("step 1 OK: .cu files parse under the CUDA shim (syntax and types only)")


def step2_cpu_extension() -> None:
    import torch

    print(f"step 2: building CPU extension with extra_cflags={host_cflags()}", flush=True)
    ext = load_extension(with_cuda=False, verbose=False)
    ext.reset_counters()
    means = torch.tensor([[8.0, 8.0]], dtype=torch.float64)
    chol = torch.tensor([[0.2, 0.0, 0.2]], dtype=torch.float64)
    opac = torch.zeros(1, dtype=torch.float64)
    colors = torch.ones(1, 3, dtype=torch.float64)
    one = torch.tensor([0, 1], dtype=torch.int32)
    out = ext.forward(means, chol, opac, colors, one, one[:1], one, one[:1], 16, 16)
    assert out[0].shape == (16, 16, 3) and float(out[0].max()) > 0.0
    print(f"step 2 OK: extension built and ran; counters={ext.get_counters()}")


def step3_nvcc() -> int:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("nvcc absent: GPU compile deferred to cluster")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        for cu in CUDA_SOURCES:
            obj = Path(tmp) / (cu.stem + ".o")
            run([nvcc, "-std=c++17", "-O2", "-DWITH_CUDA", *SM80, "-c", str(cu), "-o", str(obj)])
    load_extension(with_cuda=True, verbose=True)
    print("step 3 OK: nvcc sm_80 compile and CUDA extension build")
    return 0


def main() -> int:
    try:
        step1_mock_syntax()
        step2_cpu_extension()
    except subprocess.CalledProcessError as err:
        print(f"FAILED: {err}")
        return 1
    return step3_nvcc()


if __name__ == "__main__":
    sys.exit(main())
