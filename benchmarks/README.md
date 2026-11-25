# External model corpus

Real model source files downloaded from public GitHub repositories, used to
stress-test NetPrism outside the hand-written samples.

- Location: `benchmarks/external/<owner>__<repo>/...`
- Download method: raw GitHub files through `https://ghproxy.net/...` (direct
  GitHub was unreachable from this machine); no model weights are downloaded.
- Execution policy: the corpus is scanned with `--no-execute` by default. The
  four dependency-light trusted repos are also traced in
  `tests/test_external_corpus.py`.
- This directory is for analysis only; upstream licenses remain with the
  original repositories.

## Repositories

| Domain | Source |
|---|---|
| CNN / ResNet / DenseNet / MobileNet / EfficientNet | https://github.com/kuangliu/pytorch-cifar |
| U-Net segmentation | https://github.com/milesial/Pytorch-UNet |
| GPT | https://github.com/karpathy/nanoGPT · https://github.com/karpathy/minGPT |
| ViT / CaiT / CrossFormer / NaViT | https://github.com/lucidrains/vit-pytorch |
| Swin Transformer | https://github.com/microsoft/Swin-Transformer |
| CLIP (multimodal) | https://github.com/openai/CLIP |
| SAM (segmentation) | https://github.com/facebookresearch/segment-anything |
| DETR (detection) | https://github.com/facebookresearch/detr |
| YOLOv5 components | https://github.com/ultralytics/yolov5 |
| GCN | https://github.com/tkipf/pygcn |
| PointNet++ | https://github.com/yanx27/Pointnet_Pointnet2_pytorch |
| Mamba / SSM | https://github.com/state-spaces/mamba |
| RWKV | https://github.com/BlinkDL/RWKV-LM |
| KAN | https://github.com/KindXiaoming/pykan |
| Diffusion U-Net | https://github.com/lucidrains/denoising-diffusion-pytorch |
| NeRF | https://github.com/yenchenlin/nerf-pytorch |

## Commands

```powershell
# Static scan of every downloaded repo (no code execution)
python scripts\eval_external.py --json benchmarks\external_results.json

# High-fidelity trace for trusted dependency-light repos
python -m netprism scan benchmarks\external\kuangliu__pytorch-cifar --out out_cifar
python -m netprism scan benchmarks\external\milesial__Pytorch-UNet --out out_unet
python -m netprism scan benchmarks\external\karpathy__nanoGPT --out out_nanogpt
python -m netprism scan benchmarks\external\lucidrains__vit-pytorch --out out_vit
```

Full static/trace numbers and known limitations: [EXTERNAL.md](EXTERNAL.md).
