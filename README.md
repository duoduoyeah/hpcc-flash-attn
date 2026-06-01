# hpcc-flash-attn

Prebuilt **flash-attn** wheel(s) compiled **on the UCR HPCC**, so you can `pip install` flash-attn in
seconds instead of a ~30–60 min source build — and because the *official* wheels **don't work here**.

## Why this repo exists

The cluster's GPU nodes run **Rocky 8 (glibc 2.28)**. Every flash-attn wheel published on PyPI and on
Dao-AILab's GitHub releases is linked against **glibc ≥ 2.32**, so on these nodes `import flash_attn` fails:

```
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found (required by flash_attn_2_cuda…so)
```

The only fix is to **compile flash-attn from source on the cluster** (it then links against the host
glibc 2.28). That takes ~30–60 min of `nvcc`. This repo hosts that build **once**, as a GitHub release
asset, so every future clone / teammate just downloads and installs the wheel.

## Supported GPUs

The wheel embeds **SASS cubins for `sm_80` and `sm_120` only** (no PTX, so there is **no JIT fallback**
to other architectures):

| GPU | compute | UCR nodes | supported? |
|---|---|---|:--|
| **RTX PRO 6000 Blackwell** (`blackwell6000`) | sm_120 | gpu13, gpu14 | ✅ **validated** |
| **A100** | sm_80 | gpu06–08 | ✅ native sm_80 cubin |
| **RTX 6000 Ada** (`ada6000`) | sm_89 | gpu09, gpu10 | ✅ runs the sm_80 cubin (8.x is forward-compatible) |
| H100 | sm_90 | gpu11 | ❌ no sm_90 cubin/PTX — rebuild with `90` in the arch list |
| P100 / K80 | sm_60 / sm_37 | gpu01–05 | ❌ unsupported by flash-attn |

**Validated:** loads and decodes Qwen3-32B at tensor-parallel **`TP=4`** on RTX PRO 6000 Blackwell
(gpu13/gpu14) — smoke-tested end to end.

## The wheel

| | |
|---|---|
| package | `flash-attn==2.8.3` |
| file | `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` |
| Python | **cp312** (3.12) |
| torch | **2.9.1+cu128** |
| CUDA / compiler | cuda 12.8 · gcc 11.5.0 |
| GPU archs (SASS) | **sm_80 + sm_120** |
| built on | Rocky 8, glibc 2.28 |
| sha256 | `62599ba770b7bac08e1b431518ce958daa16088a1b9ea3bb1a3ec655e3edfe1f` |

## Install

Requires a matching environment: **Python 3.12 + `torch==2.9.1+cu128`**, a Rocky 8 / glibc 2.28 node,
and a GPU in the supported list above.

```bash
pip install https://github.com/duoduoyeah/hpcc-flash-attn/releases/download/v2.8.3-cp312-cu128/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
# or with uv:
uv pip install https://github.com/duoduoyeah/hpcc-flash-attn/releases/download/v2.8.3-cp312-cu128/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

## Rebuild (different Python / torch / GPU archs)

The wheel is specific to **(cp312, torch 2.9.1, cu128, glibc 2.28, sm_80+sm_120)**. For a different
Python or torch version — or to add an arch like H100 `sm_90` — rebuild on a cluster node and upload the
result as a new release asset:

```bash
module load cuda/12.8 gcc/11.5.0
# FLASH_ATTENTION_FORCE_BUILD=TRUE forces a real source compile; otherwise flash-attn's setup.py just
# re-downloads the unusable glibc-2.32 wheel. FLASH_ATTN_CUDA_ARCHS picks the SASS targets.
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="80;90;120" MAX_JOBS=12 NVCC_THREADS=1 \
  pip wheel --no-binary flash-attn "flash-attn==2.8.3" --no-build-isolation --no-deps -w wheelhouse/
```

(Built from the `experiments/build_flash_attn_wheel.sh` recipe in the nano-vllm project.)
