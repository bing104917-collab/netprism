# 算法分类表 · ALGORITHM_TAXONOMY

> 数据与代码：`netprism/data/algorithm_kb.json`（本文件表格均由它生成，改 JSON 后重新生成）、
> 加载器 `netprism/algorithm_kb.py`（纯 stdlib：`load_knowledge()` / `families()` / `components()` /
> `family_index()` / `component_index()` / `validate_knowledge()`）。
>
> 规模：**102 个家族**、**56 个组件**。分类器（`netprism/algorithms.py`）
> 只按「结构签名 + 别名 + 组件」打分，不为任何模型硬编码计算图。
>
> SPEC-ALGO 的 15 个 canonical family id 均可用：`cnn`/`vgg`/`resnet`/`densenet`/`mobilenet`/
> `mobilenet_v2`/`inception`/`unet`/`transformer_encoder`/`transformer_decoder`/`vit`/`mlp`/`rnn`/`hybrid`，
> `unknown` 由分类器在无证据时兜底（不入库）。具体家族（`unet_plus_plus`、`swin`、`mamba`…）证据更多，优先于通用家族。

## 1. 领域总览

| domain | 家族数 | 家族 id |
|---|---|---|
| `vision`（视觉） | 54 | `cnn`, `lenet`, `alexnet`, `vgg`, `resnet`, `resnext`, `wide_resnet`, `densenet`, `squeezenet`, `mobilenet`, `mobilenet_v2`, `mobilenet_v3`, `shufflenet_v1`, `shufflenet_v2`, `efficientnet`, `regnet`, `convnext`, `inception`, `xception`, `senet`, `cbam`, `sknet`, `repvgg`, `ghostnet`, `fcn`, `unet`, `unet_plus_plus`, `deeplabv3`, `pspnet`, `hrnet`, `segformer`, `fpn`, `panet`, `bifpn`, `yolo`, `ssd`, `faster_rcnn`, `mask_rcnn`, `retinanet`, `detr`, `vit`, `deit`, `swin`, `beit`, `mae`, `mlp_mixer`, `convmixer`, `gmlp`, `poolformer`, `mobilevit`, `coatnet`, `levit`, `crnn`, `sam` |
| `sequence`（序列/语言） | 20 | `rnn`, `lstm`, `gru`, `seq2seq`, `transformer_encoder`, `transformer_decoder`, `bert`, `gpt`, `t5`, `llama`, `moe`, `transformer_xl`, `performer`, `linformer`, `wav2vec2`, `conformer`, `mamba`, `rwkv`, `retnet`, `s4` |
| `generative`（生成） | 9 | `gan`, `dcgan`, `stylegan`, `cyclegan`, `vae`, `vqvae`, `diffusion_unet`, `dit`, `normalizing_flow` |
| `graph`（图） | 5 | `gcn`, `gat`, `graphsage`, `gin`, `mpnn` |
| `3d`（3D） | 5 | `pointnet`, `pointnet2`, `dgcnn`, `unet3d`, `nerf` |
| `multimodal`（多模态） | 3 | `whisper`, `clip`, `blip` |
| `other`（其他） | 6 | `kan`, `capsule`, `neural_ode`, `tabnet`, `mlp`, `hybrid` |

## 2. 家族分类表

`required_patterns` 是判定该家族必须命中的组件；`signature_ops` 是图节点 kind 的多重集指纹；
别名只作辅助加分，结构证据优先。

### 2.1 视觉（vision）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| CNN (`cnn`) | `conv_bn_act` | `conv`, `norm`, `activation`, `pool`, `linear` | cnn, convnet, conv net |
| LeNet-5 (`lenet`) | `plain_conv`, `classification_head` | `conv`, `activation`, `pool`, `linear` | lenet, lenet5, lenet-5 |
| AlexNet (`alexnet`) | `plain_conv`, `classification_head` | `conv`, `activation`, `pool`, `dropout`, `linear` | alexnet, alex net, alexnet features |
| VGG (`vgg`) | `plain_conv` | `conv`, `norm`, `activation`, `pool`, `linear` | vgg, vgg16, vgg19 |
| ResNet (`resnet`) | `residual`, `conv_bn_act` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | resnet, res net, resnet18 |
| ResNeXt (`resnext`) | `residual`, `bottleneck` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | resnext, res next, resnext50 |
| WideResNet (`wide_resnet`) | `residual` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | wide_resnet, wide resnet, wideresnet |
| DenseNet (`densenet`) | `dense_connection` | `conv`, `norm`, `activation`, `concat`, `pool`, `linear` | densenet, dense net, densenet121 |
| SqueezeNet (`squeezenet`) | `inception` | `conv`, `activation`, `concat`, `pool`, `linear` | squeezenet, squeeze net, squeezenet1_0 |
| MobileNet (`mobilenet`) | `depthwise_separable` | `conv`, `norm`, `activation`, `pool`, `linear` | mobilenet, mobilenetv1, mobilenet_v1 |
| MobileNetV2 (`mobilenet_v2`) | `inverted_residual` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | mobilenetv2, mobilenet_v2, mobilenet 2.0 |
| MobileNetV3 (`mobilenet_v3`) | `inverted_residual`, `se` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | mobilenetv3, mobilenet_v3, mobilenetv3_large |
| ShuffleNetV1 (`shufflenet_v1`) | `depthwise_separable` | `conv`, `norm`, `activation`, `concat`, `reshape`, `pool`, `linear` | shufflenet, shufflenetv1, shufflenet_v1 |
| ShuffleNetV2 (`shufflenet_v2`) | `depthwise_separable`, `concat_fusion` | `conv`, `norm`, `activation`, `concat`, `reshape`, `pool`, `linear` | shufflenetv2, shufflenet_v2, shufflenet 2 |
| EfficientNet (`efficientnet`) | `inverted_residual`, `se` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | efficientnet, efficient net, efficientnet_b0 |
| RegNet (`regnet`) | `residual` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | regnet, regnetx, regnety |
| ConvNeXt (`convnext`) | `convnext_block` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | convnext, conv next, convnext_tiny |
| Inception / GoogLeNet (`inception`) | `inception` | `conv`, `norm`, `activation`, `concat`, `pool`, `linear` | inception, googlenet, inception_v3 |
| Xception (`xception`) | `depthwise_separable`, `residual` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | xception, xception41, xception65 |
| SENet (`senet`) | `se` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | senet, se_resnet, se-resnet |
| CBAM (`cbam`) | `cbam` | `conv`, `norm`, `activation`, `pool`, `linear`, `custom` | cbam, cbam_resnet, convolutional block attention module |
| SKNet (`sknet`) | `sk` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | sknet, sknet50, selective kernel |
| RepVGG (`repvgg`) | `residual`, `plain_conv` | `conv`, `norm`, `activation`, `add`, `pool`, `linear` | repvgg, rep vgg, repvgg_a0 |
| GhostNet (`ghostnet`) | `inception` | `conv`, `activation`, `concat`, `add`, `pool`, `linear` | ghostnet, ghost net, ghostnetv2 |
| FCN (`fcn`) | `plain_conv`, `segmentation_head` | `conv`, `activation`, `pool`, `deconv`, `upsample`, `custom` | fcn, fully convolutional network, fcn32s |
| UNet (`unet`) | `unet_skip` | `conv`, `deconv`, `norm`, `activation`, `pool`, `concat`, `upsample` | unet, u net, u-net |
| UNet++ (`unet_plus_plus`) | `unet_skip`, `dense_connection` | `conv`, `norm`, `activation`, `pool`, `concat`, `upsample`, `deconv` | unet++, unet_plus_plus, unetpp |
| DeepLabV3 / ASPP (`deeplabv3`) | `atrous_conv`, `global_context` | `conv`, `norm`, `activation`, `pool`, `upsample`, `concat`, `linear` | deeplab, deeplabv3, deeplab v3 |
| PSPNet (`pspnet`) | `global_context` | `conv`, `activation`, `pool`, `concat`, `upsample`, `linear` | pspnet, psp net, pyramid scene parsing |
| HRNet (`hrnet`) | `plain_conv`, `concat_fusion` | `conv`, `norm`, `activation`, `add`, `concat`, `upsample`, `pool`, `linear` | hrnet, hrnetv2, hrnet_w18 |
| SegFormer (`segformer`) | `transformer_block`, `patch_embed` | `conv`, `reshape`, `attention`, `ffn`, `norm`, `linear`, `upsample`, `pool` | segformer, seg former, mix transformer |
| FPN (`fpn`) | `fpn` | `conv`, `add`, `upsample`, `pool`, `linear`, `activation` | fpn, feature pyramid network, fpn50 |
| PANet (`panet`) | `pafpn` | `conv`, `add`, `concat`, `upsample`, `pool`, `linear` | panet, pan net, pafpn |
| BiFPN (`bifpn`) | `bifpn` | `conv`, `add`, `upsample`, `pool`, `linear`, `activation` | bifpn, bi-fpn, weighted bidirectional fpn |
| YOLO (`yolo`) | `detection_head` | `conv`, `norm`, `activation`, `add`, `concat`, `pool`, `upsample`, `linear` | yolo, yolov3, yolov4 |
| SSD (`ssd`) | `detection_head` | `conv`, `activation`, `pool`, `concat`, `linear` | ssd, single shot multibox detector, ssd300 |
| Faster R-CNN (`faster_rcnn`) | `detection_head` | `conv`, `norm`, `activation`, `pool`, `linear`, `reshape`, `custom` | faster rcnn, faster_rcnn, fasterrcnn |
| Mask R-CNN (`mask_rcnn`) | `detection_head`, `segmentation_head` | `conv`, `norm`, `activation`, `pool`, `linear`, `deconv`, `upsample` | mask rcnn, mask_rcnn, maskrcnn |
| RetinaNet (`retinanet`) | `fpn`, `detection_head` | `conv`, `activation`, `add`, `upsample`, `pool`, `linear` | retinanet, retina net, focal loss |
| DETR (`detr`) | `transformer_block`, `cross_attention` | `conv`, `attention`, `ffn`, `norm`, `add`, `embedding`, `positional`, `linear` | detr, detection transformer, de tection transformer |
| ViT (`vit`) | `patch_embed`, `transformer_block` | `conv`, `reshape`, `linear`, `attention`, `ffn`, `norm`, `activation`, `add`, `pool` | vit, vision transformer, vit_b_16 |
| DeiT (`deit`) | `patch_embed`, `transformer_block` | `conv`, `reshape`, `linear`, `attention`, `ffn`, `norm`, `activation`, `add`, `pool` | deit, data efficient image transformer, deit_tiny |
| Swin Transformer (`swin`) | `window_attention` | `conv`, `reshape`, `norm`, `attention`, `ffn`, `linear`, `add`, `pool` | swin, swin transformer, swin_t |
| BEiT (`beit`) | `patch_embed`, `transformer_block` | `conv`, `reshape`, `embedding`, `attention`, `ffn`, `norm`, `linear`, `add` | beit, beit_base, bert pretraining image transformers |
| MAE (`mae`) | `patch_embed`, `transformer_block` | `conv`, `reshape`, `attention`, `ffn`, `norm`, `linear`, `add`, `deconv`, `upsample` | mae, masked autoencoder, vit mae |
| MLP-Mixer (`mlp_mixer`) | `patch_embed`, `ffn` | `conv`, `reshape`, `linear`, `norm`, `activation`, `add`, `pool` | mlp mixer, mlp-mixer, mixer |
| ConvMixer (`convmixer`) | `patch_embed`, `depthwise_separable` | `conv`, `activation`, `norm`, `add`, `pool`, `reshape` | convmixer, conv mixer, convmixer_1024_20 |
| gMLP (`gmlp`) | `gated_mlp` | `linear`, `activation`, `norm`, `reshape`, `add`, `pool`, `conv` | gmlp, g mlp, gated mlp |
| PoolFormer (`poolformer`) | `patch_embed` | `conv`, `pool`, `linear`, `norm`, `activation`, `add` | poolformer, pool former, metaformer |
| MobileViT (`mobilevit`) | `transformer_block`, `depthwise_separable` | `conv`, `norm`, `activation`, `attention`, `ffn`, `linear`, `add`, `reshape`, `pool` | mobilevit, mobile vit, mobilevit_xs |
| CoAtNet (`coatnet`) | `conv_bn_act`, `transformer_block` | `conv`, `norm`, `activation`, `attention`, `ffn`, `linear`, `add`, `pool` | coatnet, coat net, co at net |
| LeViT (`levit`) | `patch_embed`, `self_attention` | `conv`, `norm`, `attention`, `ffn`, `linear`, `activation`, `add`, `pool` | levit, levit_128, lightweight vision transformer |
| CRNN (`crnn`) | `plain_conv`, `rnn_block` | `conv`, `pool`, `activation`, `rnn`, `reshape`, `linear`, `custom` | crnn, convolutional recurrent network, ocr crnn |
| SAM (`sam`) | `patch_embed`, `transformer_block`, `cross_attention` | `conv`, `reshape`, `attention`, `ffn`, `norm`, `linear`, `activation`, `pool`, `upsample` | sam, segment anything, segment anything model |

### 2.2 序列/语言（sequence）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| RNN (`rnn`) | `rnn_block` | `rnn`, `linear`, `activation`, `embedding`, `custom` | rnn, vanilla rnn, recurrent neural network |
| LSTM (`lstm`) | `rnn_block`, `lstm_gate` | `rnn`, `linear`, `activation`, `embedding`, `custom` | lstm, long short term memory, lstmcell |
| GRU (`gru`) | `rnn_block`, `gru_gate` | `rnn`, `linear`, `activation`, `embedding`, `custom` | gru, gated recurrent unit, grucell |
| Seq2Seq (`seq2seq`) | `rnn_block` | `rnn`, `embedding`, `linear`, `attention`, `custom`, `softmax` | seq2seq, sequence to sequence, encoder decoder |
| Transformer Encoder (`transformer_encoder`) | `transformer_block`, `self_attention`, `ffn` | `attention`, `ffn`, `linear`, `norm`, `activation`, `add`, `embedding`, `positional` | transformer encoder, transformerencoder, encoder layer |
| Transformer Decoder (`transformer_decoder`) | `transformer_block`, `cross_attention` | `attention`, `ffn`, `linear`, `norm`, `activation`, `add`, `embedding`, `positional` | transformer decoder, transformerdecoder, decoder layer |
| BERT (`bert`) | `transformer_block`, `self_attention` | `embedding`, `positional`, `attention`, `ffn`, `norm`, `linear`, `activation`, `add`, `pool` | bert, bert base, bert large |
| GPT (`gpt`) | `transformer_block`, `self_attention` | `embedding`, `positional`, `attention`, `ffn`, `norm`, `linear`, `activation`, `add`, `softmax` | gpt, gpt2, gpt-2 |
| T5 (`t5`) | `transformer_block`, `cross_attention` | `embedding`, `attention`, `ffn`, `norm`, `linear`, `add`, `activation`, `softmax` | t5, text to text transfer transformer, mt5 |
| LLaMA (`llama`) | `transformer_block`, `rmsnorm`, `swiglu` | `embedding`, `attention`, `ffn`, `norm`, `linear`, `activation`, `add`, `softmax` | llama, llama2, llama3 |
| MoE (`moe`) | `moe` | `linear`, `ffn`, `norm`, `activation`, `add`, `softmax`, `custom` | moe, mixture of experts, sparse moe |
| Transformer-XL (`transformer_xl`) | `transformer_block`, `self_attention` | `embedding`, `attention`, `ffn`, `norm`, `linear`, `add`, `positional`, `softmax` | transformer xl, transformerxl, xlnet |
| Performer (`performer`) | `self_attention` | `attention`, `ffn`, `linear`, `norm`, `activation`, `add`, `softmax`, `custom` | performer, fast attention, favor+ |
| Linformer (`linformer`) | `self_attention` | `attention`, `ffn`, `linear`, `norm`, `activation`, `add`, `reshape` | linformer, linear transformer, low rank attention |
| wav2vec 2.0 (`wav2vec2`) | `transformer_block`, `conv_bn_act` | `conv`, `norm`, `activation`, `attention`, `ffn`, `linear`, `add` | wav2vec, wav2vec2, hubert |
| Conformer (`conformer`) | `transformer_block`, `conv_bn_act` | `conv`, `norm`, `activation`, `attention`, `ffn`, `linear`, `add` | conformer, convolutional transformer, speech conformer |
| Mamba (`mamba`) | `ssm` | `linear`, `conv`, `activation`, `norm`, `add`, `custom` | mamba, selective ssm, s6 |
| RWKV (`rwkv`) | `ssm` | `linear`, `activation`, `norm`, `add`, `embedding`, `custom` | rwkv, receptance weighted key value, rwkv4 |
| RetNet (`retnet`) | `self_attention` | `linear`, `activation`, `norm`, `add`, `custom`, `softmax` | retnet, retentive network, retention |
| S4 (`s4`) | `ssm` | `linear`, `activation`, `norm`, `custom`, `conv` | s4, s4d, structured state space |

### 2.3 生成（generative）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| GAN (`gan`) | `plain_conv`, `upsampling` | `conv`, `activation`, `norm`, `linear`, `deconv`, `upsample`, `custom` | gan, generative adversarial network, adversarial net |
| DCGAN (`dcgan`) | `upsampling`, `normalization` | `conv`, `deconv`, `norm`, `activation`, `linear`, `custom` | dcgan, deep convolutional gan, dcgan64 |
| StyleGAN (`stylegan`) | `ffn`, `normalization`, `upsampling` | `linear`, `conv`, `norm`, `activation`, `upsample`, `custom` | stylegan, stylegan2, stylegan3 |
| CycleGAN (`cyclegan`) | `residual` | `conv`, `deconv`, `norm`, `activation`, `add`, `custom` | cyclegan, cycle gan, pix2pix |
| VAE (`vae`) | `vae_reparam` | `conv`, `linear`, `activation`, `norm`, `upsample`, `deconv`, `custom` | vae, variational autoencoder, beta vae |
| VQ-VAE (`vqvae`) | `vq_codebook` | `conv`, `embedding`, `linear`, `activation`, `norm`, `deconv`, `custom` | vqvae, vq vae, vector quantized vae |
| Diffusion UNet (`diffusion_unet`) | `unet_skip` | `conv`, `norm`, `activation`, `concat`, `upsample`, `attention`, `ffn`, `embedding`, `add`, `linear` | diffusion, ddpm, ddim |
| DiT (`dit`) | `patch_embed`, `transformer_block` | `conv`, `reshape`, `attention`, `ffn`, `norm`, `linear`, `add`, `embedding`, `positional` | dit, diffusion transformer, dit_xl |
| Normalizing Flow (`normalizing_flow`) | `normalization` | `conv`, `linear`, `activation`, `norm`, `custom` | normalizing flow, glow, realnvp |

### 2.4 图（graph）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| GCN (`gcn`) | `gnn_message_passing` | `custom`, `linear`, `norm`, `activation`, `add` | gcn, graph convolutional network, graph conv |
| GAT (`gat`) | `gnn_message_passing`, `self_attention` | `custom`, `linear`, `attention`, `activation`, `softmax`, `add` | gat, graph attention network, gatv2 |
| GraphSAGE (`graphsage`) | `gnn_message_passing` | `custom`, `linear`, `norm`, `activation`, `concat`, `add` | graphsage, graph sage, sage |
| GIN (`gin`) | `gnn_message_passing` | `custom`, `linear`, `activation`, `add`, `norm` | gin, graph isomorphism network, gin conv |
| MPNN (`mpnn`) | `gnn_message_passing` | `custom`, `linear`, `activation`, `add`, `norm` | mpnn, message passing neural network, neural message passing |

### 2.5 3D（3d）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| PointNet (`pointnet`) | `point_mlp` | `linear`, `activation`, `norm`, `pool`, `reshape` | pointnet, point net, point cloud classification |
| PointNet++ (`pointnet2`) | `point_mlp` | `linear`, `activation`, `norm`, `pool`, `reshape`, `custom` | pointnet++, pointnet2, point net plus plus |
| DGCNN (`dgcnn`) | `gnn_message_passing` | `custom`, `linear`, `activation`, `pool`, `reshape` | dgcnn, dynamic graph cnn, edgeconv |
| 3D UNet (`unet3d`) | `unet_skip` | `conv`, `deconv`, `norm`, `activation`, `pool`, `concat`, `upsample` | 3d unet, unet3d, 3dunet |
| NeRF (`nerf`) | `nerf_mlp`, `positional_encoding` | `linear`, `activation`, `norm`, `positional`, `reshape`, `custom` | nerf, neural radiance field, nerf mlp |

### 2.6 多模态（multimodal）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| Whisper (`whisper`) | `transformer_block`, `cross_attention` | `conv`, `attention`, `ffn`, `norm`, `linear`, `add`, `embedding`, `positional`, `softmax` | whisper, openai whisper, whisper large |
| CLIP (`clip`) | `transformer_block`, `projection_head` | `conv`, `reshape`, `embedding`, `attention`, `ffn`, `norm`, `linear`, `activation`, `pool` | clip, openclip, clip vit |
| BLIP (`blip`) | `transformer_block`, `cross_attention` | `conv`, `reshape`, `embedding`, `attention`, `ffn`, `norm`, `linear`, `activation`, `pool`, `custom` | blip, blip2, bootstrapping language image pretraining |

### 2.7 其他（other）

| 家族 (id) | 必需组件 required_patterns | 结构签名 signature_ops | 别名示例 |
|---|---|---|---|
| KAN (`kan`) | `kan_spline` | `linear`, `activation`, `add`, `custom` | kan, kolmogorov arnold network, pykan |
| Capsule Network (`capsule`) | `capsule_routing` | `conv`, `linear`, `activation`, `reshape`, `custom`, `softmax` | capsule, capsnet, capsule network |
| Neural ODE (`neural_ode`) | `neural_ode_solver` | `linear`, `activation`, `conv`, `norm`, `custom` | neural ode, neural ordinary differential equation, ode net |
| TabNet (`tabnet`) | `self_attention` | `linear`, `norm`, `activation`, `attention`, `add`, `reshape`, `custom` | tabnet, attentive tabular, tabular transformer |
| MLP (`mlp`) | `ffn` | `linear`, `activation`, `dropout`, `norm`, `flatten`, `softmax` | mlp, multilayer perceptron, multi layer perceptron |
| Hybrid (`hybrid`) | `transformer_block` | `conv`, `norm`, `activation`, `attention`, `ffn`, `linear`, `add`, `pool` | hybrid, hybrid network, hybrid architecture |

## 3. 组件表

组件是家族之间的共享结构单元；`required_ops` 用图节点 kind（与 `schema.KINDS` 同词表）表达最小结构证据。

| 组件 (id) | 公式 | required_ops | tags |
|---|---|---|---|
| Conv-BN-Act (`conv_bn_act`) | `y = Act(BN(Conv(x)))` | `conv`, `norm`, `activation` | conv, bn, relu, cnn |
| Plain Conv (`plain_conv`) | `y = Act(Conv(x))（无归一化）` | `conv`, `activation` | conv, cnn, vgg, plain |
| Residual (`residual`) | `y = F(x) + x` | `add`, `conv` | skip, shortcut, identity, resnet |
| Bottleneck (`bottleneck`) | `y = x + Conv1x1(Conv3x3(Conv1x1(x)))` | `conv`, `norm`, `activation`, `add` | resnet, 1x1, 3x3, expansion |
| Depthwise Separable (`depthwise_separable`) | `y = PW(DW(x))，DW 的 groups = Cin` | `conv`, `norm`, `activation` | depthwise, separable, mobilenet, groups |
| Inverted Residual (`inverted_residual`) | `y = x + PW(DW(Act(Expand(x))))` | `conv`, `add`, `activation` | inverted, expand, mobilenetv2, mbconv |
| Squeeze-and-Excitation (`se`) | `y = x * sigmoid(W2*delta(W1*GAP(x)))` | `pool`, `linear`, `activation` | squeeze, excitation, channel attention, senet |
| CBAM (`cbam`) | `y = x * sigmoid(MLP(GAP(x))) * sigmoid(Conv7x7([AvgPool; MaxPool]))` | `pool`, `linear`, `activation`, `conv` | channel, spatial, attention, cbam |
| Selective Kernel (`sk`) | `y = sum_i softmax(FC(GAP(sum_i f_i)))_i * f_i(x)` | `pool`, `linear`, `activation`, `conv` | selective kernel, multi scale, sknet |
| Inception / Multi-branch (`inception`) | `y = Concat[B_i(x)]，同输入多 kernel 分支` | `conv`, `concat` | multi scale, googlenet, concat, fire module |
| Dense Connection (`dense_connection`) | `x_l = H_l([x_0; ...; x_{l-1}])` | `concat`, `conv` | dense, feature reuse, densenet |
| FPN (`fpn`) | `P_l = Conv(C_l + Up(P_{l+1}))` | `upsample`, `add`, `conv` | pyramid, top down, lateral, fpn |
| PAFPN (`pafpn`) | `P_l = Conv(P_l + Down(P_{l-1}) + Up(P_{l+1}))` | `upsample`, `add`, `concat` | path aggregation, bottom up, panet, neck |
| BiFPN (`bifpn`) | `P = sum_i w_i*Resize(P_i) / sum_i w_i` | `upsample`, `add`, `conv` | weighted, bidirectional, efficientdet, fusion |
| UNet Skip (`unet_skip`) | `y = Dec(Concat(Up(x), x_enc))` | `concat`, `upsample`, `deconv` | skip, encoder decoder, unet, segmentation |
| Transformer Block (`transformer_block`) | `x = x + MHA(LN(x))；x = x + FFN(LN(x))` | `attention`, `ffn`, `norm`, `add` | pre norm, residual, mha, encoder layer |
| Self-Attention (`self_attention`) | `softmax(QK^T/sqrt(d_k))V，Q/K/V 同源` | `attention` | mha, qkv, same source, attention |
| Cross-Attention (`cross_attention`) | `softmax(QK^T/sqrt(d_k))V，K/V 来自另一分支` | `attention` | mha, qkv, cross, encoder decoder |
| Feed-Forward (`ffn`) | `y = W2*Act(W1*x)，隐层扩张比 4` | `linear`, `activation` | feed forward, mlp, expand, transformer |
| Gated MLP (`gated_mlp`) | `y = W2(Act(W1*x) * sigmoid(W3*x))` | `linear`, `activation` | gating, gmlp, glu, activation |
| SwiGLU (`swiglu`) | `y = W3(SiLU(W1*x) * (W2*x))` | `linear`, `activation` | gated, silu, llama, ffn |
| RMSNorm (`rmsnorm`) | `y = x / sqrt(mean(x^2)+eps) * g` | `norm` | layer norm, rms, llm, normalization |
| RoPE (`rope`) | `q_m = R(theta*m) q，旋转 Q/K 编码相对位置` | `positional` | rotary, position, llama, relative |
| ALiBi (`alibi`) | `s_ij = q_i.k_j - m*(i-j)` | `attention`, `add` | position bias, long context, alibi |
| MoE (`moe`) | `y = sum_{i in TopK} g_i(x)*E_i(x)` | `linear`, `softmax`, `custom` | expert, router, gating, sparse |
| State Space Model (`ssm`) | `h_t = Abar h_{t-1} + Bbar x_t；y_t = C h_t` | `linear`, `custom` | state space, recurrence, mamba, s4 |
| GNN Message Passing (`gnn_message_passing`) | `h_v' = U(h_v, AGG_{u in N(v)} M(h_u,h_v))` | `custom`, `linear`, `add` | graph, neighbor, aggregate, gnn |
| Point MLP (`point_mlp`) | `y_i = MLP(x_i)，所有点共享权重；末尾 max/mean 池化` | `linear`, `activation`, `pool` | point cloud, shared mlp, permutation invariant |
| NeRF MLP (`nerf_mlp`) | `F(gamma(x), gamma(d)) -> (sigma, c)` | `linear`, `activation`, `positional` | radiance field, coordinate mlp, volume rendering |
| VAE Reparameterization (`vae_reparam`) | `z = mu + sigma*eps，eps~N(0,1)` | `linear`, `custom` | latent, reparameterization, kl, vae |
| Normalization (`normalization`) | `y = (x - mu)/sqrt(var + eps)*g + b` | `norm` | bn, ln, gn, in |
| Positional Encoding (`positional_encoding`) | `p_i = sin/cos(i/10000^{2k/d}) 或可学习 E_pos` | `positional`, `embedding` | position, sinusoid, learned, sequence |
| Classification Head (`classification_head`) | `logits = Linear(Flatten/GAP(x))` | `linear`, `pool`, `flatten` | logits, classifier, head |
| Segmentation Head (`segmentation_head`) | `y = Argmax/Up(Conv1x1(x))，逐像素输出` | `conv`, `upsample`, `linear` | dense prediction, pixel, segmentation, mask |
| Detection Head (`detection_head`) | `y = [cls, box] 或 [cls, box, mask] 多头输出` | `linear`, `reshape`, `custom` | bbox, anchor, detection, multi head |
| Projection Head (`projection_head`) | `z = Linear(x) 或 MLP(x)，映射到嵌入空间` | `linear` | embedding, contrastive, projection, metric |
| Token Embedding (`token_embedding`) | `E = Embedding(token_ids, dim)` | `embedding` | token, vocab, word embedding, sequence |
| RNN Block (`rnn_block`) | `h_t = f(W h_{t-1} + U x_t)，沿时间步展开` | `rnn` | sequence, recurrence, unroll, rnn |
| LSTM Gates (`lstm_gate`) | `i,f,o,g 四门 + 细胞状态 c_t` | `rnn`, `linear`, `activation` | lstm, gate, cell state |
| GRU Gates (`gru_gate`) | `更新门 z、重置门 r 两门` | `rnn`, `linear`, `activation` | gru, gate, reset, update |
| Patch Embedding (`patch_embed`) | `x = Conv(k=s=patch)(img) -> reshape(N, patch^2*C)` | `conv`, `reshape` | patchify, vit, image token, unfold |
| Window Attention (`window_attention`) | `注意力在 MxM 局部窗口内，层间移位窗口` | `attention`, `reshape` | window, shift, swin, local attention |
| ConvNeXt Block (`convnext_block`) | `y = x + Conv1x1(GELU(Conv1x1(DW7x7(LN(x)))))` | `conv`, `norm`, `linear`, `activation` | modern cnn, ln, depthwise 7x7, convnext |
| Atrous Conv (`atrous_conv`) | `y = Conv(k, dilation=r)（空洞/膨胀卷积）` | `conv` | dilated, atrous, aspp, deeplab |
| Global Context (`global_context`) | `g = Conv(GlobalAvgPool(x))，再广播融合` | `pool`, `conv`, `upsample` | global pooling, context, pyramid, image level |
| KAN Spline Edge (`kan_spline`) | `phi(x) = sum_k c_k B_k(x) + b*SiLU(x)` | `linear`, `activation`, `custom` | spline, bspline, kolmogorov, kan |
| Capsule Routing (`capsule_routing`) | `v_j = squash(sum_i c_ij W_ij u_i)，c 动态路由` | `linear`, `reshape`, `custom` | capsule, routing, vector, squash |
| Neural ODE Solver (`neural_ode_solver`) | `h(T) = ODESolve(f, h0, t0, T)，f 为参数化向量场` | `custom`, `linear` | ode, solver, continuous depth, torchdiffeq |
| VQ Codebook (`vq_codebook`) | `z_q = e_k, k = argmin_k L2(z_e, e_k)；量化 + 码本损失` | `embedding`, `linear`, `custom` | vector quantization, codebook, vqvae |
| Weight Tying (`weight_sharing`) | `同一参数矩阵在多个节点复用（如 embedding 与 lm head）` | `embedding`, `linear` | tied weights, shared params, reuse |
| Upsampling (`upsampling`) | `y = Upsample/ConvTranspose(x, stride=s)` | `upsample`, `deconv` | upsample, deconv, decoder, interpolate |
| Concat Fusion (`concat_fusion`) | `y = Concat([x1; x2; ...], dim=1)` | `concat` | concat, fusion, multi branch, channel |
| Pre-Norm (`pre_norm`) | `x = x + F(Norm(x))，norm 在子层之前` | `norm`, `add` | pre ln, transformer, gpt, llama |
| Post-Norm (`post_norm`) | `x = Norm(x + F(x))，norm 在残差之后` | `norm`, `add` | post ln, transformer, bert, original |
| Global Pooling (`global_pool`) | `y = mean/max_{H,W}(x) -> (N,C)` | `pool` | gap, global average, pool, head |
| Timestep Embedding (`timestep_embedding`) | `e_t = MLP(SinPos(t))，FiLM/adaLN 注入各层` | `embedding`, `linear` | diffusion, timestep, film, adaln |

## 4. 分类器打分建议

1. **别名归一化**：小写、去空格/下划线/连字符后匹配 alias 或 `model.className`，命中多个家族时按最长别名加权；
   别名缺失时不得单独决定家族（`demo/dynamic_control` 这类类名无意义的图必须靠结构）。
2. **必需组件**：`required_patterns` 每命中一个加分，未命中全部则置信度打折；`optional_patterns` 只加分不减分。
3. **结构签名**：`signature_ops` 与图中 kind 直方图做 Jaccard / 余弦相似度，作为并列家族的主要区分信号
   （例如 `densenet` 的 `concat` 密度 vs `resnet` 的 `add` 密度）。
4. **参数/形状特征**（可选）：depthwise conv 的 `groups == inChannels`、bottleneck 的 1x1-3x3-1x1、
   patch_embed 的 stride == kernel，用来在同类家族间细分（MobileNetV1 vs V2、ViT vs Swin）。
5. **置信度**：三个信号都弱时回落到 `cnn` / `transformer_encoder` / `mlp` / `hybrid` 等通用家族并给低置信度，不猜冷门家族。
6. **优先级**：具体家族（`mobilenet_v2`、`unet_plus_plus`、`swin`、`mamba`…）的 `required_patterns` 更严格，
   通用家族（`cnn`、`hybrid`、`mlp`）只应在没有更强证据时胜出。

## 5. 来源

每个家族的 `references` 都带 1–2 条论文/官方文档 URL；下表是各领域代表性来源。

- **CNN 基础**：[LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) · [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) · [VGG](https://arxiv.org/abs/1409.1556) · [ResNet](https://arxiv.org/abs/1512.03385) · [DenseNet](https://arxiv.org/abs/1608.06993) · [MobileNetV2](https://arxiv.org/abs/1801.04381) · [EfficientNet](https://arxiv.org/abs/1905.11946) · [ConvNeXt](https://arxiv.org/abs/2201.03545)
- **注意力模块**：[SENet](https://arxiv.org/abs/1709.01507) · [CBAM](https://arxiv.org/abs/1807.06521) · [SKNet](https://arxiv.org/abs/1903.06586)
- **检测/分割**：[FPN](https://arxiv.org/abs/1612.03144) · [PANet](https://arxiv.org/abs/1803.01534) · [BiFPN/EfficientDet](https://arxiv.org/abs/1911.09070) · [UNet](https://arxiv.org/abs/1505.04597) · [DeepLabV3](https://arxiv.org/abs/1706.05587) · [SegFormer](https://arxiv.org/abs/2105.15203) · [DETR](https://arxiv.org/abs/2005.12872) · [YOLOv1](https://arxiv.org/abs/1506.02640)
- **Transformer 视觉**：[ViT](https://arxiv.org/abs/2010.11929) · [Swin](https://arxiv.org/abs/2103.14030) · [MAE](https://arxiv.org/abs/2111.06377) · [MLP-Mixer](https://arxiv.org/abs/2105.01601) · [ConvMixer](https://arxiv.org/abs/2201.09792) · [PoolFormer/MetaFormer](https://arxiv.org/abs/2111.11418)
- **序列/语言**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [BERT](https://arxiv.org/abs/1810.04805) · [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [T5](https://arxiv.org/abs/1910.10683) · [LLaMA](https://arxiv.org/abs/2302.13971) · [MoE](https://arxiv.org/abs/1701.06538) · [Performer](https://arxiv.org/abs/2009.14794) · [Linformer](https://arxiv.org/abs/2006.04768)
- **生成**：[GAN](https://arxiv.org/abs/1406.2661) · [DCGAN](https://arxiv.org/abs/1511.06434) · [StyleGAN](https://arxiv.org/abs/1812.04948) · [VAE](https://arxiv.org/abs/1312.6114) · [VQ-VAE](https://arxiv.org/abs/1711.00937) · [DDPM](https://arxiv.org/abs/2006.11239) · [DiT](https://arxiv.org/abs/2212.09748)
- **状态空间/线性序列**：[S4](https://arxiv.org/abs/2110.13985) · [Mamba](https://arxiv.org/abs/2312.00752) · [RWKV](https://arxiv.org/abs/2305.13048) · [RetNet](https://arxiv.org/abs/2307.08621)
- **图/3D**：[GCN](https://arxiv.org/abs/1609.02907) · [GAT](https://arxiv.org/abs/1710.10903) · [GraphSAGE](https://arxiv.org/abs/1706.02216) · [GIN](https://arxiv.org/abs/1810.00826) · [PointNet](https://arxiv.org/abs/1612.00593) · [PointNet++](https://arxiv.org/abs/1706.02413) · [NeRF](https://arxiv.org/abs/2003.08934)
- **多模态/语音/其他**：[CLIP](https://arxiv.org/abs/2103.00020) · [BLIP](https://arxiv.org/abs/2201.12086) · [Whisper](https://arxiv.org/abs/2212.04356) · [KAN](https://arxiv.org/abs/2404.19756) · [CapsNet](https://arxiv.org/abs/1710.09829) · [Neural ODE](https://arxiv.org/abs/1806.07366)
