# Clean-Prior Residual Mask 引导的吸盘视觉重建计划书

## 1. 目标

本计划验证一个核心假设：

> 只用 `clean.png` 训练得到的 clean-prior 自监督重建模型，可以识别 suction 图中“不符合 clean 图像分布”的吸盘伪影区域；该 residual mask 能作为辅助条件，提升 `suction -> clean` 监督重建模型的恢复质量。

该方案不把 clean-prior 模型作为最终重建器，而是把它作为“异常区域提示器”或“吸盘伪影 mask 生成器”。最终重建仍由有监督网络完成：

```text
suction -> frozen clean-prior model -> prior_hat
mask = residual(suction, prior_hat)
concat(suction, mask) -> restoration model -> clean
```

## 2. 数据设定

当前数据规模：

- 188 个 group
- 188 张 clean 图
- 1885 张 suction 图
- 总 PNG 数 2073
- 总数据约 1.436 GB
- 每组 suction 平均约 10.03 张，主要为 10 张，也存在 8、9、11、20 张的变长情况

必须按 group 或 version 划分数据，避免同一个 group 的 suction 图和 clean 图泄漏到不同 split。

建议两类验证设置：

1. `group split`：每个 version 内按 group 划分 train/val/test。
2. `leave-one-version-out`：用 5 个 version 训练，剩余 1 个 version 测试，用于验证跨环境泛化。

## 3. 总体技术路线

```text
Phase 0: 数据审计与评价协议
Phase 1: clean-prior 自监督模型
Phase 2: residual mask 生成与质量验证
Phase 3: 单图 mask-guided restoration
Phase 4: 多观测 mask-guided fusion
Phase 5: diffusion / generative refiner
Phase 6: 消融、作弊检查与部署评估
```

各阶段之间的关系：

- Phase 1 产出 frozen clean-prior model。
- Phase 2 用 frozen clean-prior model 产出训练、验证、部署阶段一致可用的 mask。
- Phase 3 训练单图监督重建模型，验证 mask 是否有效。
- Phase 4 继承 Phase 3 的 encoder/backbone/loss/mask 生成方式，扩展到多 suction 输入。
- Phase 5 继承 Phase 3 或 Phase 4 的粗重建结果，作为 diffusion refiner 的条件输入。

## 4. Phase 0：数据审计与评价协议

### 4.1 要做的事

- 生成 manifest：记录 `version`, `group`, `clean_path`, `suction_paths`, `suction_count`, `metadata`。
- 检查缺失文件、重复文件、异常尺寸、异常 suction 数量。
- 自动生成有效圆形视场 mask，避免黑边主导 PSNR/SSIM。
- 固定一组可视化样例，用于跨实验对比。

### 4.2 验收指标

- manifest 中 group 数、clean 数、suction 数与磁盘统计一致。
- 所有 split 按 group 隔离，没有同 group 泄漏。
- 每个实验能输出固定格式结果：
  - input suction
  - generated mask
  - prediction
  - target clean
  - error map
- 指标至少包含：
  - masked PSNR
  - masked SSIM
  - LPIPS
  - suction-affected region error

## 5. Phase 1：Clean-Prior 自监督模型

### 5.1 训练目标

只用训练集的 `clean.png` 训练自监督重建模型，使模型学习 clean 图像流形。

可尝试三类模型：

| 模型 | 输入 | 目标 | 作用 |
|---|---|---|---|
| Denoising AutoEncoder | 加噪/模糊后的 clean | clean | 简单稳定，适合第一版 |
| MAE / ConvMAE | 随机 mask 后的 clean | clean | 学 clean 图像先验和结构补全 |
| Restoration AutoEncoder | 退化增强后的 clean | clean | 更接近后续复原网络 |

### 5.2 推荐第一版

第一版建议从 denoising autoencoder 或轻量 MAE 开始，不直接上复杂大模型。

训练输入：

```text
clean_aug = degrade(clean)
prior_hat = P(clean_aug)
loss_prior = L1(prior_hat, clean) + SSIM loss
```

`degrade(clean)` 可以包含：

- random crop / resize
- gaussian blur
- color jitter
- additive noise
- random erase
- patch mask

不要在 Phase 1 中使用 suction 图的 clean 标签配对训练，否则 clean-prior model 会变成 supervised restoration model，失去“clean-only prior”的意义。

### 5.3 验收指标

clean validation set 上：

- masked PSNR 明显高于输入退化图。
- masked SSIM 明显高于输入退化图。
- clean 图输入模型时，residual 应低且平滑。
- suction 图输入模型时，residual 应集中在吸盘轮廓、反光、遮挡和形变区域。

第一轮不建议设置绝对阈值，可以使用相对验收：

- `prior_hat(clean_aug)` 相比 `clean_aug` 至少提升 masked PSNR。
- clean 图上的平均 residual 显著低于 suction 图上的平均 residual。
- 人工查看固定样例时，mask 能覆盖主要吸盘伪影，而不是只响应纹理边缘。

## 6. Phase 2：Residual Mask 生成

### 6.1 基础 mask

冻结 clean-prior model `P`，对 suction 生成：

```text
prior_hat = P(suction)
diff_rgb = abs(suction - prior_hat)
diff_gray = mean(diff_rgb, channel)
mask = normalize(diff_gray)
```

推荐第一版：

```text
mask = GaussianBlur(diff_gray)
mask = percentile_normalize(mask, p_low=5, p_high=95)
mask = clamp(mask, 0, 1)
```

### 6.2 可选增强 mask

如果基础 mask 太碎，可以加入梯度项：

```text
diff_pixel = mean(abs(suction - prior_hat), channel)
diff_grad = mean(abs(grad(suction) - grad(prior_hat)), channel)
mask = normalize(alpha * diff_pixel + beta * diff_grad)
```

如果 mask 覆盖过大，可以加入 soft threshold：

```text
mask = sigmoid((mask - threshold) / temperature)
```

### 6.3 验证阶段如何获取 mask

验证和部署阶段不能访问 clean。mask 必须使用同一条路径生成：

```text
suction -> frozen P -> prior_hat -> residual mask
```

不推荐用历史平均 mask 作为主方法。历史平均只能作为 baseline：

```text
mask_avg[index] = average residual mask for suction_index on train set
```

它可以测试“吸盘位置先验”是否有效，但不能适配当前图像中的反光、遮挡和局部形变。

### 6.4 验收指标

- train/val/test 的 mask 生成过程完全一致，均不访问 clean。
- mask 在 clean 图上的平均响应低于 suction 图。
- mask 不应大面积覆盖黑边背景。
- mask 可视化能稳定覆盖 suction 伪影核心区域。
- 使用 mask 后的 restoration 网络，在 suction-affected region error 上优于 no-mask baseline。

## 7. Phase 3：单图 Mask-Guided Restoration

### 7.1 基础任务

```text
input:  suction_i, mask_i
target: clean
output: pred_clean
```

推荐输入格式：

```text
concat_input = [suction_rgb, mask_1ch]
pred = R(concat_input)
```

### 7.2 mask 接入方式

优先尝试以下三种：

| 方案 | 做法 | 复杂度 | 优先级 |
|---|---|---:|---:|
| Input concat | 把 mask 作为第 4 通道输入 | 低 | 最高 |
| Weighted loss | mask 区域提高 loss 权重 | 低 | 最高 |
| Feature gating | 用 mask 调制 encoder/skip feature | 中 | 第二轮 |

推荐第一版同时使用 input concat 和 weighted loss：

```text
pred = R(concat(suction, mask))
loss = L1(pred, clean) * (1 + lambda * mask)
     + ssim_loss(pred, clean)
     + perceptual_loss(pred, clean)
```

注意：loss 里的 mask 也必须来自 frozen prior residual，而不是 `abs(suction - clean)`。

### 7.3 候选重建模型

第一批模型：

| 模型 | 理由 | 预期 |
|---|---|---|
| U-Net / ResUNet | 简单、稳定、小数据友好 | 快速验证方案是否成立 |
| NAFNet-style restoration net | 轻量、适合图像复原 | 速度和质量平衡 |
| Restormer | 图像复原强 baseline | 更好处理长程依赖 |
| Uformer | transformer restoration baseline | 对结构恢复可能更强 |
| SwinIR | 窗口注意力，纹理恢复较好 | 作为高质量对照 |

MAE 不放在这个表的原因是：这里的模型需要直接执行 `suction + mask -> clean` 的密集像素复原。vanilla MAE 是自监督预训练框架，不是直接的监督复原网络。但 MAE 产出的 encoder 可以作为这些 restoration 网络的初始化。

### 7.4 对照实验

必须包含：

1. Identity baseline：`pred = suction`
2. No-mask restoration：`R(suction) -> clean`
3. Mask input only：`R(concat(suction, mask)) -> clean`
4. Mask loss only：`R(suction) -> clean`，loss 加权
5. Mask input + mask loss

### 7.5 验收指标

进入下一阶段的最低标准：

- No-mask restoration 明显优于 identity baseline。
- Mask-guided restoration 在 suction-affected region error 上优于 no-mask restoration。
- full-FOV masked PSNR/SSIM 不应明显下降。
- 固定样例可视化中，mask-guided 模型不能产生明显 hallucination 或颜色漂移。

建议相对验收：

- `R(suction)` 相比 identity baseline 有稳定提升。
- `R(suction, mask)` 相比 `R(suction)` 在遮挡/反光区域有稳定提升。
- 至少两个不同 version 的验证结果趋势一致。

## 8. Phase 4：多观测 Mask-Guided Fusion

### 8.1 目标

验证多张 suction 图是否提供互补信息。

```text
{suction_i, mask_i, index_i}_{i=1..N} -> clean
```

### 8.2 推荐架构

```text
for each i:
  feat_i = shared_encoder(concat(suction_i, mask_i, index_embedding_i))

fused_feat = aggregation({feat_i})
pred_clean = decoder(fused_feat)
```

聚合方式从简单到复杂：

1. mean/max pooling
2. attention pooling
3. Set Transformer
4. cross-attention decoder

训练时随机采样输入张数：

```text
N in {1, 2, 4, 8, all}
```

这样模型能适应真实场景中 suction 数量变化。

### 8.3 验收指标

- `N=1` 时性能不低于 Phase 3 同类模型。
- `N=2/4/8` 随输入数增加，masked PSNR/SSIM 或 region error 有单调或近似单调改善。
- 变长输入下模型输出稳定，不依赖固定 10 张输入。
- attention 可视化或输入 ablation 能说明模型确实利用了不同 suction 观测。

## 9. Phase 5：Diffusion / Generative Refiner

### 9.1 使用前提

只有当 Phase 3 或 Phase 4 已有可靠 deterministic reconstruction 后，再进入 diffusion refiner。不要让 diffusion 从 suction 直接自由生成 clean。

推荐输入：

```text
condition = {
  suction,
  residual_mask,
  coarse_pred
}
diffusion_refiner(condition) -> refined_clean
```

### 9.2 候选方案

| 方案 | 用法 | 风险 |
|---|---|---|
| Latent diffusion refiner | 在 latent 空间细化 coarse_pred | 可能 hallucinate |
| Image-to-image diffusion | 以 coarse_pred 为初始图 | 推理慢 |
| Rectified flow / consistency model | 更快生成式细化 | 实现复杂度较高 |

### 9.3 验收指标

- LPIPS 或人工观感优于 deterministic coarse_pred。
- masked PSNR/SSIM 不能明显下降。
- 不能在抓取相关区域生成与输入矛盾的结构。
- 推理时间必须满足实际抓取部署约束；如果不满足，只能作为离线上限实验。

## 10. 作弊与泄漏检查

### 10.1 明确禁止

- 用 `abs(suction - clean)` 生成训练输入 mask。
- 用验证/测试集 clean 训练 clean-prior model。
- 同一个 group 的 clean/suction 出现在不同 split。
- 训练时使用真值 mask，验证时使用估计 mask。

### 10.2 允许

- 用训练集 clean 训练 clean-prior model。
- 训练、验证、测试阶段都用 `suction -> frozen P -> mask` 生成 mask。
- 在 loss 中使用 clean 作为监督目标。
- 用训练集统计历史平均 mask 作为 baseline，但不能作为主评估方案。

### 10.3 验收指标

- 所有实验日志记录 mask 来源。
- 每个 split 的 group 列表保存到文件。
- 训练和验证使用同一份 mask 生成代码。
- ablation 中包含 no-mask baseline，证明提升不是评价偏差。

## 11. 推荐实验顺序

第一轮最小可行实验：

1. 生成 manifest 和 split。
2. 训练 clean-prior denoising autoencoder。
3. 生成 residual mask，并可视化 20 组样例。
4. 训练 ResUNet no-mask baseline。
5. 训练 ResUNet mask input + weighted loss。
6. 对比 identity、no-mask、mask-guided 三者。

第二轮模型扩展：

1. 替换 clean-prior 为 MAE / ConvMAE。
2. 替换 restoration 为 Restormer / Uformer / SwinIR。
3. 比较不同 mask 计算方式。
4. 做 leave-one-version-out 泛化测试。

第三轮多观测：

1. shared encoder + mean pooling。
2. shared encoder + attention pooling。
3. 输入数量消融：1、2、4、8、all。

第四轮生成式细化：

1. 以最佳 deterministic 模型输出作为 coarse_pred。
2. 训练 diffusion refiner。
3. 评估质量、幻觉风险和推理速度。

## 12. 最终交付物

每个阶段应产出：

- 训练配置文件
- split 文件
- checkpoint
- 指标表
- 固定样例可视化
- error map
- mask 可视化
- 实验结论

最终报告应回答：

1. clean-prior residual mask 是否能稳定定位 suction 伪影？
2. mask-guided restoration 是否优于 no-mask restoration？
3. MAE clean-prior 是否优于简单 denoising autoencoder？
4. 多观测融合相比单图是否有实际收益？
5. diffusion refiner 是否带来真实收益，还是主要产生幻觉？
6. 哪个模型组合最适合实际抓取部署？

## 13. 推荐路线结论

当前最推荐的主线是：

```text
Clean-only DAE/MAE prior
  -> residual mask
  -> ResUNet/Restormer mask-guided restoration
  -> multi-view attention fusion
  -> optional diffusion refiner
```

这条路线能把 MAE 的 clean 图像先验、多观测互补信息和监督复原网络结合起来，同时保持训练-验证-部署阶段的一致性，降低使用 clean 生成 mask 带来的作弊风险。
