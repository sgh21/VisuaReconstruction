# 吸盘观测模糊视觉重建：任务分析与探索计划

## 1. 背景与问题本质

本项目的数据是典型的配对视觉重建数据：每个环境版本 `dataset_v*` 下包含多个 `group_*`，每个 group 有一个清晰目标图 `clean.png`，以及多张吸盘在不同位置观测得到的 `suction_*.png`。任务目标是在真实抓取时，从透过吸盘得到的模糊观测中恢复接近 `clean.png` 的清晰视觉结果。

从抽样图像看，suction 输入不是普通的均匀退焦模糊。它同时包含透明吸盘轮廓、反光、遮挡、局部变形、暗角和可能的位置相关折射。因此该任务应界定为“带物理伪影的条件图像复原/去遮挡/多观测融合”，而不只是 classic deblurring。

## 2. 数据与输入输出界定

当前数据结构：

```text
dataset/
  dataset_v1/
    metadata.jsonl
    group_0001/
      clean.png
      suction_00.png
      suction_01.png
      ...
  dataset_v2/
  ...
```

已抽样确认图像分辨率为 `1920x1080`。每个 group 通常约 10 张 suction 输入，但存在变长输入，因此模型设计最好天然支持可变数量观测。

建议定义三种任务形态：

1. 单观测重建：输入一张 `suction_i`，输出同 group 的 `clean`。这是实际部署最简单的版本，也是基础 baseline。
2. 多观测融合重建：输入同一 group 的若干张 `suction_i`，输出 `clean`。不同吸盘位置提供互补信息，理论上上限更高。
3. 在线增量重建：随着抓取过程拿到第 1 到第 t 张 suction 图，持续输出更好的 clean 估计。这更贴近真实机器人流程。

形式化定义：

```text
single: f_theta(x_{v,g,i}, m_{v,g,i}) -> y_{v,g}
multi:  f_theta({x_{v,g,i}, m_{v,g,i}}_{i in S}) -> y_{v,g}
```

其中 `x` 是 suction 图，`y` 是 clean 图，`m` 是可选元数据，例如 suction index、focus、exposure、frame_id 等。

## 3. 监督信号与损失设计

这是有配对监督的 image-to-image restoration：

- 输入：`suction_*.png`
- 目标：同 group 的 `clean.png`
- 基础损失：L1 或 Charbonnier，比 L2 更适合复原任务。
- 结构损失：SSIM / MS-SSIM，用于保持局部结构。
- 感知损失：LPIPS 或 VGG perceptual loss，用于减少过度平滑。
- 边缘损失：Sobel / Laplacian gradient loss，可帮助恢复纹理和边界。
- Masked loss：建议对有效圆形视场加权，降低黑边区域主导指标的风险。

如果后续能获得吸盘位置或可见性 mask，可以增加局部加权：对吸盘遮挡区、反光区、物体边界区赋予更高权重。

## 4. 现有经典视觉重建框架是否适用

### 4.1 传统去模糊/超分框架

普通 deblurring 假设模糊核相对稳定，但本任务中吸盘引入的是空间变化、透明遮挡、反射和折射。传统去模糊可以作为弱 baseline，但不应作为主方案。

更合适的是现代 supervised image restoration 网络，例如 U-Net/ResUNet、Restormer、Uformer、SwinIR。这类模型直接学习 `suction -> clean` 的映射，训练和评估路径清晰，适合作为第一阶段主 baseline。

### 4.2 MAE

MAE 的优势在于自监督预训练和缺失区域建模。它可以帮助模型学习 clean 图的结构先验，也可以用于大规模无标签 suction/clean 图预训练。

但 MAE 的标准任务是随机 patch mask 后重建原图，和本项目的真实退化过程不一致。直接套 MAE 不如监督式 restoration 直接。更合理的用法是：

- 用 MAE/ViT encoder 做预训练初始化；
- 构造 suction-aware masking，把吸盘遮挡/反光区域当作结构化缺失；
- 在监督训练阶段使用 MAE encoder + restoration decoder。

结论：MAE 适合作为预训练或辅助，不适合作为第一版核心解法。

### 4.3 VAE

VAE 能学习清晰图像的低维潜变量先验，也能输出不确定性。但 VAE 的重建通常偏平滑，容易损失纹理和边缘，对本任务这种需要高频细节的视觉重建不占优。

合理用法包括：

- 学 clean 图 latent prior；
- 作为低维表征或异常检测模块；
- 为 diffusion 或 latent restoration 提供压缩空间。

结论：VAE 不建议作为主 baseline，可作为生成先验实验。

### 4.4 Diffusion

条件 diffusion 很适合强退化、遮挡和多解复原，尤其当 suction 遮挡导致信息缺失时，diffusion 可以借助图像先验补全。但它也有明显代价：

- 数据量小时容易过拟合或幻觉；
- 推理速度慢，机器人实时抓取可能受限；
- 像素保真与真实可用性需要严格评估。

更稳妥的路线是把 diffusion 放在第二阶段：

1. 先训练确定性 restoration 模型得到可靠 baseline；
2. 再训练 conditional latent diffusion 或 diffusion refiner；
3. 使用强条件约束和 masked loss，避免生成不符合 suction 观测的细节。

结论：diffusion 有潜力，但不应先于确定性 baseline。

## 5. 推荐网络训练架构

### 5.1 第一阶段：单图监督复原 baseline

输入一张 suction 图，输出 clean 图：

```text
suction RGB -> encoder-restoration-decoder -> clean RGB
```

候选模型：

- ResUNet：实现简单，适合小数据集。
- Restormer / Uformer：更贴近图像复原，适合捕获长程依赖。
- SwinIR：窗口注意力对纹理复原较强。

建议从较低分辨率开始，例如 `512x288` 或围绕圆形视场裁剪后 resize，再逐步提升到更高分辨率。

### 5.2 第二阶段：多 suction 融合模型

多张 suction 图具有互补信息，应作为核心方向：

```text
每张 suction_i -> shared encoder -> feature_i
feature_i + index embedding -> set/attention aggregation -> decoder -> clean
```

关键设计：

- shared encoder 保证不同 suction 输入共享表征；
- index embedding 编码 `suction_00`、`suction_01` 等位置顺序；
- aggregation 支持变长输入，可用 mean/max pooling、attention pooling、Set Transformer 或 cross-attention；
- 训练时随机采样输入张数，增强模型对真实场景中输入数量变化的鲁棒性。

### 5.3 第三阶段：生成式 refiner

当 deterministic 模型达到稳定 baseline 后，再考虑 diffusion：

```text
suction/multi-fusion condition + coarse reconstruction -> latent diffusion refiner -> clean
```

这里 diffusion 不负责从零恢复，而是修复残留模糊、反光和纹理缺失。这样更可控，也更适合小数据条件。

## 6. 训练与数据划分

必须避免同一 group 泄漏到训练和测试中，因为同一 group 的所有 suction 图共享同一个 clean 目标。

建议至少设置两类划分：

1. Group split：每个 version 内按 group 划分 train/val/test。
2. Version split：leave-one-version-out，用 5 个 version 训练，剩余 1 个 version 测试。

推荐消融：

- 单图 vs 多图；
- 输入数量 `1/2/4/8/all`；
- 是否使用 suction index embedding；
- 是否使用 metadata；
- full-image loss vs circular-FOV masked loss；
- L1 only vs L1 + SSIM + perceptual。

## 7. 评估指标

基础指标：

- PSNR：像素保真，但容易被黑边影响。
- SSIM / MS-SSIM：结构相似性。
- LPIPS：感知质量。

建议指标：

- Masked PSNR / Masked SSIM：只在有效圆形视场内计算。
- Edge error：评估边界和纹理恢复。
- Region-specific metric：对 suction 遮挡区或高反光区单独评估。
- Visual grid：固定展示 input、prediction、target、error map。
- Downstream grasp metric：如果最终用于抓取，应让重建质量与抓取成功率或抓取点估计误差挂钩。

## 8. 探索计划

### Phase 0：数据审计

- 生成 manifest：记录 version、group、clean path、suction paths、metadata。
- 统计每组 suction 数量、图像尺寸、缺失文件、重复路径。
- 自动估计有效圆形视场 mask。
- 输出固定样例可视化网格。

产物：`manifest.csv/jsonl`、数据统计报告、样例图。

### Phase 1：低成本 baseline

- identity baseline：直接把 suction 当输出，测出最低线。
- per-group suction mean/median：观察多观测平均是否能减少反光或遮挡。
- simple ResUNet single-image baseline。

目标：确认监督映射是否可学、损失和 mask 是否合理。

### Phase 2：主线 deterministic restoration

- 训练 Restormer/Uformer/SwinIR 类模型。
- 加入 masked loss、SSIM、perceptual loss。
- 比较不同输入分辨率和裁剪策略。

目标：得到稳定、可复现实验基线。

### Phase 3：多观测融合

- shared encoder + attention aggregation。
- 训练时随机采样不同数量 suction 输入。
- 做输入数量消融，判断多观测带来的边际收益。

目标：验证“吸盘不同位置观测提供互补信息”这一核心假设。

### Phase 4：MAE/自监督预训练

- 使用 clean 和 suction 图做 MAE 风格预训练。
- 将 encoder 初始化到 restoration 模型。
- 比较小数据条件下是否提升泛化。

目标：判断 MAE 对跨 version 泛化是否有效。

### Phase 5：Diffusion refiner

- 在 deterministic 输出基础上训练 conditional latent diffusion/refiner。
- 强约束输出与目标 clean 对齐，重点观察遮挡区和纹理区。
- 评估推理速度是否满足抓取部署。

目标：判断 diffusion 是否带来实质质量提升，而不是只产生视觉上更锐但不真实的细节。

### Phase 6：部署与任务闭环

- 压缩模型或蒸馏，评估实时性。
- 接入抓取任务指标。
- 设计失败案例库：强反光、吸盘遮挡目标、极暗视场、跨 version 光照变化。

目标：让重建模型服务于实际抓取，而不是只优化离线图像指标。

## 9. 当前建议结论

该任务可以套入现有视觉重建框架，但不能简单等同于普通 MAE、VAE 或 deblurring。最合理的路线是：

1. 先按监督式 image-to-image restoration 建立确定性 baseline；
2. 再扩展到多 suction 变长输入融合；
3. 用 MAE 做预训练增强，而不是直接作为主模型；
4. 用 diffusion 做后期 refiner 或强遮挡补全实验；
5. 始终用 masked 指标和抓取相关指标约束模型，防止生成式方法产生不可用幻觉。
