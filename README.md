# hpcc-flash-attn

Prebuilt **flash-attn** wheels compiled **on the UCR HPCC** (Rocky 8 / glibc 2.28), for nodes where
the official PyPI / Dao-AILab release wheels **fail to load** (they require glibc ≥ 2.32). Built once
from source so future installs are a fast `pip install <url>` instead of a ~1 h `nvcc` compile.

## Wheel

| | |
|---|---|
| package | `flash-attn==2.8.3` |
| file | `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` |
| Python | **cp312** (3.12) |
| torch | **2.9.1+cu128** |
| CUDA | 12.8 (`cuda/12.8` module) · gcc 11.5.0 |
| GPU archs | **sm_80** (A100, RTX 6000 Ada) + **sm_120** (RTX PRO 6000 Blackwell) |
| built on | Rocky 8, glibc 2.28 |
| sha256 | `62599ba770b7bac08e1b431518ce958daa16088a1b9ea3bb1a3ec655e3edfe1f` |

**Validated:** loads and decodes Qwen3-32B at tensor-parallel `TP=4` on RTX PRO 6000 Blackwell
(`gpu13/gpu14`, sm_120) — smoke-tested end to end.

## Install

Match the pins above (Python 3.12 + `torch==2.9.1+cu128`, on a Rocky 8 / glibc 2.28 node):

```bash
pip install https://github.com/duoduoyeah/hpcc-flash-attn/releases/download/v2.8.3-cp312-cu128/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
# or with uv:
uv pip install https://github.com/duoduoyeah/hpcc-flash-attn/releases/download/v2.8.3-cp312-cu128/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

## How it was built

`FLASH_ATTENTION_FORCE_BUILD=TRUE` + `--no-binary flash-attn` to force a real source compile (the
default would download the glibc-2.32 release wheel), against `cuda/12.8` + `gcc/11.5.0` with
`FLASH_ATTN_CUDA_ARCHS="80;120"`, single-threaded `nvcc` to bound RAM. See the build script in the
nano-vllm project (`experiments/build_flash_attn_wheel.sh`).

## Other Python/torch combos

The wheel is specific to (cp312, torch 2.9.1, cu128, glibc 2.28). For a different Python or torch,
rebuild on the cluster and add the new wheel as another release asset.
