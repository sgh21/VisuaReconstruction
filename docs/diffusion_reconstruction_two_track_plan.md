# 吸盘视觉重建的 Diffusion 两阶段方案计划书

## 1. 目标与基本判断

最终目标是学习从有吸盘观测图 `suction_*.png` 恢复无吸盘遮挡的目标图 `clean.png`：

```text
input:  suction observation
output: clean scene
```

当前数据规模约为：

- 188 个 group
- 188 张 clean 目标图
- 1885 张 suction 输入图

这个规模不适合从零训练完整 diffusion。更合理的方向是使用官方/主流预训练 diffusion 模型，只训练较小的任务适配模块，或者先用确定性模型得到粗重建，再让 diffusion 做 refiner。

本文给出两个可落地版本：

1. **版本 A：基础重建模型 + Diffusion Refiner**
2. **版本 B：微调预训练 Diffusion / ControlNet**

推荐优先做版本 A，因为它更稳、更容易验证，不容易让 diffusion 产生不受控幻觉。版本 B 作为更生成式、更强表达能力的路线。

参考官方资源：

- Hugging Face Diffusers image-to-image 文档：https://huggingface.co/docs/diffusers/en/using-diffusers/img2img
- Hugging Face Diffusers ControlNet 训练文档：https://huggingface.co/docs/diffusers/en/training/controlnet
- Hugging Face Diffusers ControlNet 使用文档：https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet
- SDXL base 1.0 model card：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- SDXL refiner 1.0 model card：https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

## 2. 公共数据与评估设定

### 2.1 预处理

所有图像统一走当前项目 dataloader 预处理：

```text
PIL RGB
-> center crop 1080x1080
-> resize to model size
-> tensor in [0, 1]
```

建议 diffusion 阶段使用两种分辨率：

- 快速验证：`256x256`
- 正式实验：`512x512`

由于当前 clean/suction 视场本身是圆形，指标必须避免被黑边主导。

### 2.2 数据划分

必须按 group 隔离：

```text
train group / val group / test group
```

不允许同一 group 的 clean 和 suction 出现在不同 split。

推荐两类 split：

1. **Group split**：所有 version 混合后按 group 划分。
2. **Leave-one-version-out**：用 5 个 version 训练，剩下 1 个 version 测试。

### 2.3 通用指标

基础指标：

- masked PSNR
- masked SSIM
- LPIPS
- L1 / Charbonnier
- residual error map

视觉验收：

- suction 输入
- coarse prediction
- diffusion output
- clean target
- error map
- suction artifact mask / overlay

任务验收重点：

- 吸盘轮廓是否被去除
- 高光反射是否被抑制
- 木纹/目标边界是否保真
- 是否生成不存在的纹理
- 抓取相关区域是否稳定

## 3. 版本 A：基础重建模型 + Diffusion Refiner

### 3.1 核心思想

先训练一个确定性模型完成主要映射：

```text
suction -> coarse_clean
```

再训练 diffusion refiner 修复残余伪影：

```text
condition = suction + coarse_clean + optional mask
diffusion_refiner(condition) -> refined_clean
```

这条路线把 diffusion 限制在“局部细化”和“遮挡补全”角色，避免它从随机噪声自由生成整张图。

### 3.2 阶段 A1：确定性基础模型

输入输出：

```text
input:  suction RGB
target: clean RGB
output: coarse_clean RGB
```

候选模型：

- U-Net / ResUNet
- NAFNet-style restoration net
- Restormer
- Uformer
- SwinIR

优先推荐：

```text
ResUNet -> NAFNet/Restormer
```

因为当前数据量小，先要确认 `suction -> clean` 映射是否稳定可学。

训练损失：

```text
L_base =
  L1(coarse, clean)
  + 0.2 * SSIM_loss(coarse, clean)
  + 0.1 * gradient_loss(coarse, clean)
```

如果已有 clean-prior residual mask：

```text
L_base_weighted = L_base * (1 + lambda * artifact_mask)
```

但 mask 只能由 `suction + frozen prior model` 生成，不能用 `abs(suction - clean)` 作为训练输入。

阶段验收：

- coarse_clean 明显优于直接输入 suction。
- masked PSNR / SSIM 高于 identity baseline。
- suction artifact 区域 error 明显下降。
- 不出现严重颜色漂移。

### 3.3 阶段 A2：Diffusion Refiner

Diffusion 不从随机噪声生成完整 clean，而是在 coarse prediction 附近做细化。

推荐形式：

```text
x0 = clean
condition = concat(suction, coarse_clean, artifact_mask)
xt = add_noise(clean, t)
model_input = xt + condition + timestep
model_target = noise epsilon
```

训练目标：

```text
epsilon ~ N(0, I)
xt = sqrt(alpha_t) * clean + sqrt(1-alpha_t) * epsilon
epsilon_pred = D(xt, t, condition)
L_diff = MSE(epsilon_pred, epsilon)
```

推理：

```text
given suction
coarse_clean = base_model(suction)
artifact_mask = prior_or_residual_mask(suction)
start from noisy latent around coarse_clean
denoise under condition
output refined_clean
```

推荐从 latent diffusion 做，而不是 pixel diffusion：

```text
VAE.encode(clean) -> latent_clean
VAE.encode(coarse_clean) -> latent_coarse
diffusion refines latent
VAE.decode(refined_latent) -> refined_clean
```

### 3.4 可用预训练底座

第一版建议使用 Stable Diffusion 1.5 或 SDXL 的 VAE/UNet 作为底座。SDXL 的官方 model card 说明它是 latent diffusion，并且有 base/refiner 两段式设计，这和“先 coarse 后 refiner”的思路相近。

建议优先级：

1. Stable Diffusion 1.5：更轻，512x512，训练成本低。
2. SDXL base/refiner：质量更高，但显存和训练成本更大。

当前 RTX 5070 12GB 更适合先试 SD 1.5 / 512x512 / fp16 / LoRA 或小 adapter。

### 3.5 版本 A 的优点和风险

优点：

- 最稳定。
- diffusion 只负责细化，幻觉风险较低。
- deterministic baseline 可以单独部署。
- 数据量小也能推进。

风险：

- 上限受基础模型 coarse_clean 限制。
- 如果 coarse_clean 已经错误，refiner 可能强化错误。
- 需要维护两个模型。

### 3.6 版本 A 实验矩阵

第一轮：

| 实验 | Base | Refiner | Condition | 目标 |
|---|---|---|---|---|
| A0 | identity | none | none | 下限 |
| A1 | ResUNet | none | suction | 基础监督复原 |
| A2 | Restormer/NAFNet | none | suction | 强 baseline |
| A3 | best base | latent diffusion refiner | coarse | 判断 diffusion 是否提升 |
| A4 | best base | latent diffusion refiner | coarse + suction | 判断 suction condition 是否必要 |
| A5 | best base | latent diffusion refiner | coarse + suction + mask | 判断 mask 是否有效 |

验收：

- A3/A4/A5 的 LPIPS 和视觉质量优于 base。
- masked PSNR/SSIM 不能明显低于 base。
- 人工检查不能出现抓取相关区域幻觉。

## 4. 版本 B：微调预训练 Diffusion / ControlNet

### 4.1 核心思想

直接学习条件生成：

```text
p(clean | suction)
```

把 suction 当作条件图，而不是把 suction 当成标准 diffusion 的高斯噪声。

训练形式：

```text
target image: clean
condition image: suction
noise target: epsilon
```

标准 diffusion 训练：

```text
latent_clean = VAE.encode(clean)
epsilon ~ N(0, I)
zt = add_noise(latent_clean, epsilon, t)
epsilon_pred = UNet(zt, t, condition=suction)
loss = MSE(epsilon_pred, epsilon)
```

### 4.2 方案 B1：ControlNet 微调

ControlNet 的设计是冻结原 diffusion 大模型，在旁路训练条件控制分支。Diffusers 官方文档说明 ControlNet 是在预训练模型之上增加 adapter，用额外输入图像控制生成；这种方式适合小数据任务，因为主体模型可以冻结。

输入输出：

```text
condition: suction RGB 或 suction + mask
target: clean RGB
base model: Stable Diffusion 1.5 或 SDXL
trainable: ControlNet
frozen: VAE, text encoder, base UNet
```

Prompt 设计：

因为任务不是文本生成，prompt 应保持固定、简单：

```text
"a clean camera view without suction cup occlusion"
```

或者空 prompt / domain prompt，并固定整个训练和测试。

条件图设计：

第一版：

```text
condition_image = suction
```

第二版：

```text
condition_image = concat_or_rgb_encode(suction, artifact_mask)
```

如果 ControlNet 只接受 3 通道，可以做：

```text
R,G,B = suction RGB
或
R = suction gray
G = residual mask
B = edge map
```

但第一版建议先用 suction RGB，避免过早引入编码选择。

训练目标：

```text
clean as generated target
suction as condition
MSE noise prediction loss
```

推理：

```text
given suction
fixed prompt
ControlNet(condition=suction)
sample clean
```

### 4.3 方案 B2：LoRA 微调 UNet

LoRA 只在预训练 UNet 的 attention 或卷积模块中插入低秩参数。

输入：

```text
initial image: suction or blank/noise
prompt: fixed clean scene prompt
target: clean
```

优点：

- 参数少。
- 对小数据更友好。
- 训练和存储成本低。

缺点：

- 只靠 LoRA 不一定能强约束 suction 的空间结构。
- 对图像条件的利用通常不如 ControlNet 明确。

因此 LoRA 更适合作为第二阶段，不建议优先于 ControlNet。

### 4.4 方案 B3：Img2Img / SDEdit 风格微调

Diffusers image-to-image 文档中，img2img 的基本过程是：输入初始图像，编码为 latent，加噪，再由 diffusion 去噪得到新图。

本任务可以定义为：

```text
initial image: suction
target image: clean
strength: 控制从 suction 偏离多少
```

这种方式最接近用户直觉中的“把 suction 去噪成 clean”。

但风险是：

- strength 低：吸盘残留。
- strength 高：结构漂移或幻觉。
- 没有显式监督微调时，不会自动知道“吸盘应该被去掉”。

因此 img2img 更适合做推理方式或 refiner，而不是唯一训练方案。

### 4.5 版本 B 实验矩阵

| 实验 | 方法 | Trainable | Condition | 分辨率 | 目标 |
|---|---|---|---|---|---|
| B1 | ControlNet SD1.5 | ControlNet only | suction RGB | 512 | 主推荐 |
| B2 | ControlNet SD1.5 | ControlNet only | suction + mask encoding | 512 | 看 mask 是否提升 |
| B3 | LoRA SD1.5 | LoRA only | prompt + img2img | 512 | 低成本对照 |
| B4 | SDXL ControlNet | ControlNet only | suction RGB | 512/768 | 高质量上限 |
| B5 | full UNet finetune | UNet | suction | 512 | 不推荐，只作为上限 |

优先顺序：

```text
B1 -> B2 -> B3 -> B4
```

不建议先做 B5，因为当前数据量太小，过拟合和幻觉风险很高。

### 4.6 版本 B 验收标准

必须满足：

- 输出与 clean 的圆形视场位置一致。
- 吸盘伪影被去除。
- 物体边缘和木纹不出现明显凭空生成。
- 对同一 group 不同 suction index 输出稳定。
- leave-one-version-out 不崩溃。

如果出现：

- 输出过于“自然图像化”
- 木纹变成随机纹理
- 抓取区域结构改变
- 同一输入多次采样差异过大

则说明 diffusion 幻觉风险过高，需要加强 condition 或退回版本 A。

## 5. 两个版本的对比

| 维度 | 版本 A：Base + Refiner | 版本 B：微调 Diffusion |
---|---|---|
| 稳定性 | 高 | 中低 |
| 数据需求 | 较低 | 较高 |
| 幻觉风险 | 较低 | 较高 |
| 实现复杂度 | 中 | 高 |
| 可解释性 | 高 | 中 |
| 推理速度 | 较快 | 较慢 |
| 质量上限 | 中高 | 高 |
| 推荐优先级 | 最高 | 第二 |

当前数据规模下，优先做版本 A。版本 B 可以并行做小规模可行性验证，但不应替代 deterministic baseline。

## 6. 推荐落地顺序

### Step 1：准备 paired diffusion manifest

生成每条训练样本：

```json
{
  "version": "dataset_v1",
  "group": "group_0001",
  "suction_path": ".../suction_00.png",
  "clean_path": ".../clean.png",
  "suction_index": 0
}
```

预处理：

```text
center crop 1080x1080
resize 512x512
normalize for selected model
```

### Step 2：先完成 deterministic baseline

训练：

```text
suction -> clean
```

输出：

```text
coarse_clean
artifact_mask
error_map
```

### Step 3：版本 A refiner

训练 latent diffusion refiner：

```text
condition = suction + coarse_clean + artifact_mask
target = clean
```

只接受 refiner 结果在人工视觉和 masked metrics 上同时优于 coarse baseline。

### Step 4：版本 B ControlNet 小实验

使用 SD1.5 或 SDXL：

```text
condition = suction
target = clean
train ControlNet only
```

先跑小规模：

```text
train groups: 30
val groups: 10
resolution: 512
steps: 1000-3000
```

如果视觉上能稳定去除吸盘，再扩大到全量数据。

## 7. 当前推荐结论

最推荐的主线：

```text
Restoration base model
-> residual/artifact mask
-> latent diffusion refiner
-> optional ControlNet finetune
```

不推荐当前直接从零训练 diffusion，也不推荐直接 full fine-tune SDXL UNet。

版本 A 是工程上最稳的路线，能清楚回答 diffusion 是否真正提升了重建质量。版本 B 是更有生成能力的路线，适合在版本 A 形成稳定 baseline 后探索上限。
