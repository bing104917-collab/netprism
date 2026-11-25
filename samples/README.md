# NetPrism sample models

Realistic, CPU-friendly PyTorch models used as the reference input for NetPrism.
Every file is self-contained (only `torch`), same style as a small open-source
training repo, and each class can be constructed with no arguments (except
`NeedsArgs` on purpose). All inputs/outputs below are the shapes used by the
tests and by NetPrism's tracer.

| File | Class | Input | Output | What it covers |
|---|---|---|---|---|
| `cnn_vgg.py` | `VGGMini` | `(1, 3, 32, 32)` float32 | `(1, 10)` | `nn.Sequential` CNN: 3×3 and 1×1 convs, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, linear head |
| `resnet_mini.py` | `ResNetMini` | `(1, 3, 32, 32)` float32 | `(1, 10)` | Custom `BasicBlock`, residual `add`, 1×1 projection shortcut, strided downsampling, two residual stages |
| `transformer_classifier.py` | `TinyTransformerClassifier` | `(1, 16)` int64 | `(1, 10)` | `nn.Embedding`, sinusoidal positional-encoding buffer, `nn.TransformerEncoderLayer`/`Encoder` (MHA + FFN + LayerNorm), mean pool, classifier head |
| `custom_attention.py` | `TinyGPT` | `(1, 16)` int64 | `(1, 16, 128)` | Hand-written `SelfAttention` (QKV `Linear`, scale, softmax, causal mask), `FeedForward`, LayerNorm, residual adds, LM head |
| `unet_mini.py` | `UNetMini` | `(1, 3, 32, 32)` float32 | `(1, 1, 32, 32)` | Encoder/decoder, `MaxPool2d`, `ConvTranspose2d` upsampling, `torch.cat` skip connections |
| `mlp.py` | `MLP` | `(1, 64)` float32 | `(1, 10)` | Plain linear stack with GELU / ReLU / Dropout |
| `needs_args.py` | `NeedsArgs(width)` | `(1, 64)` float32 (static only) | `(1, 10)` | No default for `width`, so it **cannot be instantiated**; NetPrism must fall back to static AST parsing |
| `dynamic_control.py` | `DynamicNet` | `(1, 3, 32, 32)` float32 | `(1, 10)` | `if x.mean() > 0` data-dependent branch: FX symbolic tracing fails on purpose, eager execution and the hook fallback work |
| `not_a_model.py` | — | — | — | Utility functions plus a `nn.Module` subclass without `forward`; the scanner must skip this file |

Notes:

- `transformer_classifier.py` and `custom_attention.py` use a **fixed sequence
  length of 16 tokens** (the positional-encoding / causal-mask size) so they
  stay `torch.fx`-traceable.
- `needs_args.py` and `dynamic_control.py` are intentionally not part of the
  default-constructible set; they exercise the static and hook fallbacks.
- All models are tiny: a forward pass on CPU takes a few milliseconds and no
  pretrained weights or downloads are involved.

## Run them manually

Bundled interpreter (not on `PATH`):

```powershell
$py = 'C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
cd <project-root>
& $py samples\mlp.py          # prints torch.Size([1, 10])
```

Or parse them with NetPrism:

```powershell
& $py -m netprism scan samples --out scratch\netprism_out
& $py -m netprism inspect samples\resnet_mini.py
```

## Tests

```powershell
& $py -m pytest tests/test_samples.py -q
```

`tests/test_samples.py` imports each file by path, checks construction, forward
shapes, parameter counts (computed from the live model), and the intentional
behaviour of `needs_args` / `dynamic_control` / `not_a_model`.
