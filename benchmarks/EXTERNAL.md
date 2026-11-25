# External corpus results

Scanned 2026-11-25 on Windows 11 / CPU / PyTorch 2.15.0+cpu. Static scans use
`--no-execute`; downloaded code is never imported or run. Counts are for the
default CLI variant folding.

## High-fidelity trace (trusted, dependency-light repos)

| Repo | models | trace | layers | params |
|---|---:|---:|---:|---:|
| kuangliu/pytorch-cifar | 10 | 10 | 1,551 | 53,331,372 |
| milesial/Pytorch-UNet | 1 | 1 | 67 | 31,038,218 |
| karpathy/nanoGPT | 1 | 1 | 113 | 124,475,904 |
| lucidrains/vit-pytorch | 5 | 5 | 727 | 29,199,538 |

## Static scan (full corpus)

| Repo | models | warnings | layers | params |
|---|---:|---:|---:|---:|
| BlinkDL__RWKV-LM | 1 | 8 | 13 | 2 |
| facebookresearch__detr | 3 | 11 | 94 | 18,915,328 |
| facebookresearch__segment-anything | 4 | 42 | 504 | 32,871,178 |
| karpathy__minGPT | 1 | 4 | 2 | 0 |
| karpathy__nanoGPT | 1 | 4 | 1 | 0 |
| KindXiaoming__pykan | 2 | 4 | 8 | 0 |
| kuangliu__pytorch-cifar | 10 | 19 | 310 | 6,667,112 |
| lucidrains__denoising-diffusion-pytorch | 2 | 10 | 99 | 1,040 |
| lucidrains__vit-pytorch | 5 | 51 | 121 | 513,306 |
| microsoft__Swin-Transformer | 1 | 0 | 10 | 384 |
| milesial__Pytorch-UNet | 1 | 1 | 10 | 64 |
| openai__CLIP | 1 | 7 | 21 | 8 |
| state-spaces__mamba | 2 | 7 | 3 | 0 |
| tkipf__pygcn | 1 | 1 | 5 | 0 |
| ultralytics__yolov5 | 8 | 9 | 40 | 601,676 |
| yanx27__Pointnet_Pointnet2_pytorch | 3 | 11 | 72 | 657,666 |
| yenchenlin__nerf-pytorch | 1 | 2 | 3 | 34,308 |

Totals: **47 models, 0 file-level errors, 191 static warnings**.

## What the static scan exposes

1. **Required constructor arguments are not synthesized in static mode.**
   `UNet(n_channels=..., n_classes=...)`, `GCN(nfeat, nhid, nclass)`,
   `CLIP(embed_dim, ...)`, `Mamba(d_model=...)` therefore keep only the module
   tree that can be built without those values; params are partial. Trace mode
   (default CLI) synthesizes them and gives the correct numbers.
2. **Custom / library ops are kept as conservative unknown shapes.** Examples:
   Mamba's `mamba_split_conv1d_scan_combined`, KAN's `coef2curve`, einops
   `rearrange`/`repeat`, SAM's dynamic window code. They produce `custom` nodes
   and `shape unknown` warnings; they are never treated as identity.
3. **Dynamic control flow** (Swin window attention, SAM `if` branches,
   `for ... in range(self.depth)`) is analyzed body-only; graphs are structural,
   not guaranteed complete.
4. **TensorFlow repos are out of scope.** The original PointNet repo is TF; the
   PyTorch PointNet++ repo is included instead.
5. **Import failures are not warnings.** Missing `torchvision`, `timm`, `scipy`
   etc. simply make the scan static; the static AST analysis is the intended
   fallback.

## Regression tests

`tests/test_external_corpus.py`:
- static scan all 17 repos: no errors, graphs pass schema, no identity fallback;
- expected model names present;
- trace checks for CIFAR (10 models), UNet (31,038,218 params), nanoGPT
  (124,475,904), ViT (5 models).

```powershell
python -m pytest tests\test_external_corpus.py -q
```

## Next targets (not yet implemented)

- einops `rearrange` / `repeat` / `reduce` shape rules;
- static constructor synthesis from parameter-name heuristics;
- `torch.var` / `Tensor.permute` / `unbind` / `zip` shape rules;
- more graph / point-cloud / detection repos with GPU/CUDA ops.
