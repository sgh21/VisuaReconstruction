# AGENTS.md

## 会话启动规则

- 在本仓库开启新会话时，必须先检查并阅读仓库根目录的 `AGENTS.md` / `AGENT.md`，再做文件扫描、代码阅读、修改或运行命令。
- 每次进行试验验证后，需要询问用户是否将试验结果更新进 `AGENTS.md`；如果用户同意，应自动维护本文件。
- 对临时测试代码，测试结束后需要询问用户是否保留；如果不保留，应清理临时代码，保持项目简洁、可读、易维护。
- 图片生成任务优先使用 ChatGPT image2，不使用本地 PIL 绘图库生成图片。

## 项目目标

项目研究“吸盘观测引起的模糊视觉重建”。每个环境 version 下有若干 group；每个 group 的 `clean.png` 是目标清晰图，其余 `suction_*.png` 是吸盘位于不同位置时透过吸盘观测得到的模糊/遮挡图。目标是在实际抓取任务中，根据透过吸盘得到的模糊观测，重建对应清晰视觉结果。

## 已观察到的数据结构

- 项目根目录：`D:\WorkSpace\VisuaReconstruction`
- 数据目录：`dataset/`
- 当前版本目录：`dataset_v1` 到 `dataset_v6`
- 每个版本下有 `group_0001` 等 group 目录，以及 `metadata.jsonl`
- 每个 group 通常包含：
  - `clean.png`
  - `suction_00.png`、`suction_01.png` 等若干 suction 输入
- 图像分辨率：已抽样确认为 `1920x1080`
- 当前统计：
  - `dataset_v1`: 37 groups, 409 files
  - `dataset_v2`: 30 groups, 332 files
  - `dataset_v3`: 30 groups, 337 files
  - `dataset_v4`: 30 groups, 328 files
  - `dataset_v5`: 30 groups, 331 files
  - `dataset_v6`: 31 groups, 342 files
- 常见 suction 数量为每组约 10 张，但存在 8、9、11、20 张等变长情况。
- `metadata.jsonl` 已观察字段包括 `ts`, `group`, `type`, `path`, `focus_set`, `exposure_set`, `focus_actual`, `exposure_actual`, `width`, `height`, `index`, `frame_id`。当前样例中 suction 位置主要可见为 `index`，尚未发现显式二维位置字段。

## 当前任务界定

- 基本输入：单张或多张 `suction_*.png`
- 基本输出：同 group 下对应的 `clean.png`
- 监督形式：有配对监督的 image-to-image restoration
- 建议优先任务形式：
  - 单观测重建：`f(x_i) -> y`
  - 多观测融合重建：`f({x_i}_{i=1..N}) -> y`
  - 在线增量重建：`f(x_1, ..., x_t) -> y_t`
- clean 图本身带圆形视场、暗角和局部遮挡，因此评估不应只看整幅全图指标，应重点考虑有效圆形视场和任务相关区域。

## 当前技术判断

- 该问题不是单纯去模糊；suction 输入包含退焦、透明吸盘结构、反光、遮挡、折射/形变以及位置相关伪影。
- 最直接的框架是监督式图像复原/条件图像翻译。
- 第一阶段应优先建立确定性 baseline，例如 U-Net/ResUNet、Restormer、Uformer、SwinIR 类 image restoration 网络。
- 可探索 clean-prior residual mask 方案：只用训练集 clean 图训练自监督重建模型，冻结后输入 suction，通过 `abs(suction - prior_hat)` 等 residual 生成 mask，再将 `suction + mask` 输入监督重建网络。训练、验证、部署阶段的 mask 都必须由 suction 和 frozen prior model 生成，不能使用 `abs(suction - clean)` 作为网络输入。
- 多张 suction 图提供互补观测，应重点探索共享编码器加注意力/集合聚合的多输入融合架构。
- MAE 更适合作为自监督预训练或特征初始化，不应作为第一版直接解法。
- VAE 可建模先验和不确定性，但高频重建容易过平滑，不适合作为主 baseline。
- Diffusion 适合处理强遮挡和多解性，可作为第二阶段高质量生成式 refiner；小数据集条件下需要谨慎控制幻觉和推理速度。

## 推荐评估方向

- 使用 masked PSNR / SSIM / LPIPS，mask 应覆盖有效圆形视场，必要时额外评估 suction 影响区域。
- 按 group 划分训练/验证/测试，避免同一 group 的 suction 图泄漏到不同 split。
- 增加 leave-one-version-out 测试，用于衡量跨环境 version 泛化。
- 记录输入数量消融：1、2、4、8、all 张 suction 输入。
- 最终应加入抓取任务相关指标或人工判读指标，避免只优化像素指标。

## 2026-07-05 Clean-prior 实验记录

- 代码状态：
  - MAE 已改为与官方 MAE ViT-B/16 asymmetric encoder-decoder 对齐，使用 `timm` 的 `PatchEmbed`/`Block`，decoder 为 8 层、512 维、16 heads。
  - MAE `weights=default` 加载官方 visualization checkpoint：`https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth`。
  - MAE 和 torchvision segmentation 模型均在加载官方权重时打印 `load_state_dict` 返回信息；当前已验证会打印 `<All keys matched successfully>`。
  - MAE 内部使用 ImageNet mean/std normalize，输出再反归一化回 `[0,1]`，避免指标和 overlay 落在归一化空间。
  - dataloader 已加入以图像中心为中心的 `1080x1080` 裁切，再 resize 到模型输入尺寸。
  - 训练日志和曲线写入各 run 目录下的 `tensorboard/`。
  - 数据与实验输出目录由 `.gitignore` 忽略；代码已推送到 `sgh21/VisuaReconstruction`，最近相关提交包括 `dca4421`, `10ce572`, `756a97e`, `6f9837a`, `ba807b1`。

- MAE clean-prior：
  - 命令参数：`model=mae_vit_b_16`, `weights=default`, `official-image-size`, `epoch=200`, `batch-size=48`, `num-workers=6`。
  - run 目录：`runs/clean_prior/mae_vit_b_16_official_e200_bs48_norm`
  - 官方权重加载日志：`Official MAE load_state_dict msg: <All keys matched successfully>`。
  - 完整训练到 epoch 200；最终 `best val_psnr=17.40`。
  - 测试结果：clean eval `mean_l1=0.12235`, `mean_psnr=17.45`。
  - suction 可视化只导出了前 16 张样例：`runs/clean_prior/mae_vit_b_16_official_e200_bs48_norm/test_outputs_limit16`。
  - 观察：MAE 作为 clean-prior 可运行，但 suction prior 图可见明显 patch/block artifacts，人工检查时需要重点看吸盘区域 overlay 是否可靠。

- LRASPP clean-prior：
  - 命令参数：`model=lraspp_mobilenet_v3_large`, `weights=default`, `official-image-size`, `batch-size=48`, `num-workers=6`。
  - run 目录：`runs/clean_prior/lraspp_mobilenet_v3_large_official_e200_bs48`
  - 官方权重加载日志：`Official torchvision load_state_dict msg for lraspp_mobilenet_v3_large: <All keys matched successfully>`。
  - 训练未跑满 200 epoch；用户要求当前巡检结束后停止，约在 epoch 105/106 附近中断，最后明确记录的 `best val_psnr=29.39`。
  - 完整测试已保存：`runs/clean_prior/lraspp_mobilenet_v3_large_official_e200_bs48/test_outputs_full`
  - 测试结果：clean eval `mean_l1=0.06573`, `mean_psnr=22.09`。
  - 输出包含全部 1885 张 suction 的 `_suction.png`, `_prior.png`, `_mask.png`, `_overlay.png`, `_grid.png` 和 CSV 指标。

- FCN clean-prior：
  - `fcn_resnet50` 使用官方 torchvision segmentation 权重，加载日志为 `Official torchvision load_state_dict msg for fcn_resnet50: <All keys matched successfully>`。
  - `batch-size=48` 在 520x520 输入下 OOM，失败目录：`runs/clean_prior/fcn_resnet50_official_e200_bs48`。
  - `batch-size=16` 可启动但显存仍接近满载，按用户要求停止，未作为主要测试结果。
  - 当前采用 `batch-size=8`, `num-workers=6`, `official-image-size`。
  - run 目录：`runs/clean_prior/fcn_resnet50_official_e200_bs8`
  - 训练未跑满 200 epoch；按用户要求在一次巡检后停止，约 epoch 32 附近中断，最后明确记录的 `best val_psnr=34.13`。
  - 完整测试已保存：`runs/clean_prior/fcn_resnet50_official_e200_bs8/test_outputs_full`
  - 测试结果：clean eval `mean_l1=0.03002`, `mean_psnr=27.87`。
  - 输出包含全部 1885 张 suction 的 `_suction.png`, `_prior.png`, `_mask.png`, `_overlay.png`, `_grid.png` 和 CSV 指标。

- 当前阶段判断：
  - 就 clean-prior 自监督重建的 clean eval 指标看，当前 FCN bs8 中断 checkpoint 明显优于 LRASPP 和 MAE。
  - MAE 虽然官方结构和权重已对齐，但在本数据 clean-prior 输出中容易出现 patch 伪影；更适合继续作为自监督先验/初始化候选，而不是直接作为最优 clean-prior baseline。
  - 后续若继续正式比较，应优先让 FCN bs8 跑满或设置统一 wall-clock/epoch budget，再对同一批 suction overlay 做人工伪影评价。

## 项目文档

- 当前任务分析与探索计划：`docs/task_analysis_and_exploration_plan.md`
- Clean-prior residual mask 引导重建计划书：`docs/clean_prior_residual_mask_plan.md`
