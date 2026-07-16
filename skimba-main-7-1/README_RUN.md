# skimba-main-7-1：Linux 命令行运行指南

本文只覆盖项目自己维护的可执行入口。`tests/`、第三方 `mamba/` 目录以及仅供导入的 Python 模块不在范围内。

## 1. 运行前准备

所有命令都应从本项目根目录执行：

```bash
cd /path/to/project-master/core-code/skimba-main-7-1
```

建议使用 Python 3.10 或更新版本。部分脚本使用了 `float | None` 这类 Python 3.10 类型语法。确认 Python 和 CUDA：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

仓库没有提供完整、独立的环境安装脚本或 requirements 文件。运行训练前至少要保证代码实际导入的 PyTorch、PyYAML、StrictYAML、NumPy、tqdm、einops、TensorBoard、spconv、torch-scatter 等依赖可用；绘图需要 matplotlib，体素可视化需要 Open3D。具体版本应与服务器的 CUDA、驱动和 GPU 架构匹配。

主要配置文件是：

```text
config/semantickitti_autoencoder.yaml
```

执行前重点检查以下配置：

- `dataset_params.data_root`：数据总目录。
- `dataset_params.kitti_root`：SemanticKITTI 图像、标定等原始数据目录。
- `dataset_params.gt_root`：稠密体素真值目录；相对路径会拼到 `data_root` 后。
- `dataset_params.vae_feature_root`：VAE latent 输出/读取目录。
- `dataset_params.image_condition_root`：图像条件特征目录。
- `dataset_params.partial_condition_root`：partial condition 特征目录。
- `dataset_params.label_mapping`：SemanticKITTI 标签映射 YAML。
- `train_params.vae_checkpoint`：语义 VAE checkpoint。
- `train_params.model_load_path`、`model_save_path`：模型读取和保存目录。

配置中现有的 `/mnt/data/...` 路径只有在你的 Linux 服务器确实存在这些目录时才可直接使用，否则必须替换。

常见数据目录结构如下：

```text
<root>/
└── sequences/
    └── 08/
        └── voxels/
            └── 000000.label 或 000000.bin
```

查看某个脚本的完整参数：

```bash
python scripts/data/select_best_semantic_vae.py --help
```

## 2. 训练 Skimba 扩散网络

入口：`train_diffusion_network_2.py`

两张 GPU 推荐使用 PyTorch DDP：

```bash
torchrun --standalone --nproc_per_node=2 train_diffusion_network_2.py \
  --config_path config/semantickitti_autoencoder.yaml
```

`train_data_loader.batch_size` 和 `num_workers` 都是每个进程（即每张 GPU）的值。全局 batch size = 每卡 batch size × GPU 数量；默认每卡 batch size 为 8，两卡全局 batch size 为 16。默认每个进程使用 8 个训练数据 worker，因此两卡合计 16 个；CPU 或系统内存压力较大时可先降至每进程 4 个。

直接用 Python 启动仍支持单卡训练：

```bash
python train_diffusion_network_2.py \
  --config_path config/semantickitti_autoencoder.yaml
```

训练数据位置、batch size、学习率、训练轮数、VAE checkpoint 和输出目录由 YAML 控制。脚本还提供 `--resume` 恢复训练，以及只导出单帧预测的 `--export-only` 参数；完整参数以 `python train_diffusion_network_2.py --help` 为准。

训练输出包括：

- `train_params.model_save_path` 下的 `best_<epoch>_<metric>.pth`；
- 可恢复训练的 `protect_<epoch>.pth`；
- `output_<epoch>.txt` 验证结果；
- `validation_predictions/` 或 `validation_output_path` 指定位置中的预测；
- 当前目录的 `logs/` TensorBoard 日志。

查看日志：

```bash
tensorboard --logdir logs --port 6006
```

首次双卡运行建议先做短程检查：确认两张卡都有持续利用率，只有 rank 0 写日志和 checkpoint，首次验证前后没有 collective hang，并将每卡峰值显存控制在约 28 GiB 以下。随后比较相同数据设置下的 samples/s；本地 CPU 测试不能替代服务器上的 NCCL、显存和吞吐验证。

注意：该脚本在解析 `--help` 前就导入训练依赖，所以缺少 PyYAML、PyTorch、spconv 等依赖时，连帮助信息也可能无法显示。

### 2.1 在 sequence 08 全部帧上重新验证最佳 diffusion checkpoint

独立入口：`scripts/evaluation/evaluate_diffusion_checkpoint.py`

从 `train_params.model_save_path` 自动扫描 `best_<epoch>_<mIoU>.pth`，按文件名中的 mIoU 选择当前最好 checkpoint，并在 sequence 08 全部帧上重新验证：

```bash
python scripts/evaluation/evaluate_diffusion_checkpoint.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --seed 20260713 \
  --device cuda:0
```

该入口会复制 validation loader 配置并移除 `frame_divisor`，固定使用 `imageset: val`、`batch_size: 1` 和 `shuffle: false`。它只加载模型并评测，不恢复 optimizer/scheduler，也不会继续训练。默认 seed 固定随机初始噪声和 timestep，方便重复比较；可以通过 `--seed` 修改。

要明确验证某个 checkpoint，而不按文件名自动选择：

```bash
python scripts/evaluation/evaluate_diffusion_checkpoint.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --checkpoint /data/checkpoints/best_590_4.674597113412418.pth \
  --seed 20260713 \
  --device cuda:0
```

默认报告目录为：

```text
<model_save_path>/full_validation/<checkpoint-stem>/seed_<seed>/
```

其中 `metrics.json` 保存 checkpoint、帧数、逐类 IoU、semantic mIoU、completion IoU、epsilon MSE 和交叉熵；`report.txt` 保存便于阅读的同类摘要。完整验证默认不导出每帧 `.label`，避免占用大量磁盘。可以用 `--output-dir` 覆盖报告目录。

## 3. 导出 MonoScene 图像条件特征

入口：`train_joint_vae_monoscene_condition.py`

虽然文件名包含 `train_joint`，当前主函数实际执行的是 MonoScene/FLoSP 图像条件特征导出，不是模型训练。

先用少量样本和 dry-run 检查路径：

```bash
python train_joint_vae_monoscene_condition.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --kitti-root /data/SemanticKITTI \
  --monoscene-root /opt/MonoScene-master \
  --sequences 08 \
  --num-samples 2 \
  --dry-run \
  --device cuda:0
```

确认无误后去掉 `--dry-run`：

```bash
python train_joint_vae_monoscene_condition.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --kitti-root /data/SemanticKITTI \
  --output-root /data/Image_transform_Voxel_Condition_Features \
  --monoscene-root /opt/MonoScene-master \
  --monoscene-checkpoint /opt/MonoScene-master/trained_models/monoscene_kitti.ckpt \
  --sequences 00,01,02,03,04,05,06,07,08,09,10 \
  --device cuda:0
```

默认读取 `vae_root/sequences/XX/voxels/*.bin`，写入 `output_root/sequences/XX/voxels/*.bin`。已存在文件默认跳过；用 `--overwrite` 覆盖。该 legacy helper 默认导出 64 通道、空间形状 `64×64×8` 的 MonoScene raw feature；当前默认扩散配置直接读取这一路 raw image condition。

旧版 8 通道 image-condition ablation 可将 MonoScene/FLoSP raw 64 通道 image condition 用 PCA/SVD 离线压缩到 8 通道：

```bash
python scripts/data/pca_condition_features_to_8.py fit-transform \
  --config-path config/semantickitti_autoencoder.yaml \
  --condition-root image \
  --input-root /data/Image_transform_Voxel_Condition_Features \
  --output-root /data/Image_transform_Voxel_Condition_Features_pca8ch \
  --pca-model /data/Image_transform_Voxel_Condition_Features_pca8ch/pca_model.npz \
  --split train \
  --max-voxels-per-file 2048 \
  --seed 20260702 \
  --out-json results/monoscene_image_pca8ch.json \
  --out-csv results/monoscene_image_pca8ch.csv
```

`fit-transform` 默认只用 SemanticKITTI train split 拟合 PCA，保存固定投影矩阵后再把同一输入根目录下的 condition 文件导出为 float32 `[8,64,64,8]`。PCA 投影等价于固定权重的 `1×1×1 Conv3d(64,8)`；它不是随机卷积，也不随 diffusion 训练。当前 YAML 默认读取 raw 64 通道：

```yaml
dataset_params:
  image_condition_root: "Image_transform_Voxel_Condition_Features"
```

## 4. 选择最佳语义 VAE checkpoint

入口：`scripts/data/select_best_semantic_vae.py`

比较目录中的全部 `best_*.pth`：

```bash
python scripts/data/select_best_semantic_vae.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-checkpoint-dir /data/checkpoints/VAE \
  --dataset-root /data/skimba_data \
  --sequences 08 \
  --num-samples 20 \
  --mode mean \
  --device cuda:0 \
  --out-json results/vae_ranking.json \
  --out-csv results/vae_ranking.csv \
  --frame-csv results/vae_ranking_frames.csv
```

只比较指定 checkpoint：

```bash
python scripts/data/select_best_semantic_vae.py \
  --vae-checkpoints /data/VAE/best_40_x.pth /data/VAE/best_80_y.pth \
  --dataset-root /data/skimba_data \
  --sequences 08 \
  --num-samples 20
```

排序优先级是：无异常标记、semantic mIoU 降序、稀有类 mIoU 降序、CE 升序、occupancy IoU 降序。脚本会在终端打印 `Recommended checkpoint`。

### 4.1 全序列 VAE 审查与版本推荐

入口：`scripts/audit/audit_vae_checkpoints.py`

下面的命令会扫描指定目录中的全部 `best_*.pth`，在数据根目录下自动发现所有序列，并从每个序列确定性抽样 20 帧：

```bash
python scripts/audit/audit_vae_checkpoints.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --checkpoint-dir /mnt/data/datasets/kitti/odometry/skimba_data/model_save_path/VAE \
  --frames-per-sequence 20 \
  --seed 20260621 \
  --output-dir /path/to/project-master-results/skimba-main-7-1/vae_audit \
  --device cuda
```

默认输出位于项目同级的 `project-master-results/skimba-main-7-1/vae_audit`，避免远程传输代码目录时重复复制大体积实验结果。服务器存储布局不同时可用 `--output-dir` 显式覆盖；项目内的 `results/` 已被 Git 忽略。

脚本只在 checkpoint 间比较 VAE 重建能力，不修改训练或 checkpoint。每个 checkpoint 都使用 posterior mean 进行确定性重建，并共享完全相同的 same frame manifest，避免因抽到不同帧而产生虚假的版本优势。少于 20 帧的序列会使用其全部帧；`--seed`、数据内容和 checkpoint 目录不变时，抽样清单不变。可用 `--sequences 00,08` 限制序列范围，或用 `--label-root`、`--label-mapping` 覆盖 YAML 路径。

输出目录包含：

- `vae_audit_report.md`：推荐版本、总排名、与 YAML 当前版本的差值，以及推荐版本最弱的序列。
- `vae_audit.json`：完整配置、抽样清单、全局/逐序列指标、异常标记和错误信息。
- `vae_audit_summary.csv`：每个 checkpoint 一行的汇总指标。
- `vae_audit_frames.csv`：每个 checkpoint、每个抽样帧一行的诊断指标。

全局和逐序列 semantic mIoU 都由累计混淆矩阵计算；CE 按有效体素数加权。排序优先级依次为：成功且无异常、semantic mIoU、稀有类 mIoU、CE、occupancy IoU。单个损坏 checkpoint 会写入报告但不会中止其他候选；没有任何候选可评估或推荐结果出现非有限 latent/logits 时返回状态码 2，否则返回 0。

## 5. 导出语义 VAE latent

入口：`scripts/data/export_z0_semantic_latents.py`

```bash
python scripts/data/export_z0_semantic_latents.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-checkpoint /data/checkpoints/VAE/best_81_x.pth \
  --dataset-root /data/skimba_data \
  --label-root /data/skimba_data/data_odometry_voxels_all \
  --output-root /data/skimba_data/VAE_Encoder_Features_Semantic20 \
  --sequences 00,01,02,03,04,05,06,07,09,10 \
  --mode mean \
  --device cuda:0
```

输出为 raw float32 文件：

```text
<output-root>/sequences/XX/voxels/<frame>.bin
```

默认 latent 形状为 `8,64,64,8`。已有文件默认跳过；重新生成时加 `--overwrite`。调试时可加 `--num-samples 2`，关闭进度条用 `--no-progress`。

### 5.1 计算 train-only 逐通道 latent 统计量

统计脚本只读取标签映射 YAML 的 `split.train` 序列，不接受 validation/test 序列覆盖参数；输入仍是磁盘上的 raw float32 posterior-mean latent，统计过程使用 float64 的逐文件 parallel-Welford 合并：

```bash
python scripts/data/compute_latent_channel_stats.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --latent-root /data/skimba_data/VAE_Encoder_Features_Semantic20 \
  --label-mapping config/label_mapping/semantic-kitti-multiscan.yaml \
  --output /data/skimba_data/latent_channel_stats.json
```

JSON 会记录 8 个通道的 mean/std、样本数、每通道及总元素数、train 序列、latent shape、posterior mean 导出模式、输入/累积 dtype、population std 定义和源路径。脚本不改写已有 latent。

启用训练时修改同一个 YAML：

```yaml
model_params:
  latent_normalization:
    enabled: True
    stats_path: "/data/skimba_data/latent_channel_stats.json"
    min_std: 1e-6
```

然后照常启动：

```bash
python train_diffusion_network_2.py \
  --config_path config/semantickitti_autoencoder.yaml
```

启用后，训练与验证都在 DDPM `add_noise` 前对 raw latent 标准化，DDIM 采样结果在 VAE decoder 前反标准化。condition feature 不参与该变换。基于 raw latent 训练的旧扩散 checkpoint 不能在启用标准化时恢复；加载器会明确报错。预训练 VAE checkpoint 不受影响。

## 6. 验证已导出的 latent

入口：`scripts/data/validate_semantic_vae_latents.py`

验证 Skimba raw `.bin` latent：

```bash
python scripts/data/validate_semantic_vae_latents.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-checkpoint /data/checkpoints/VAE/best_81_x.pth \
  --latent-root /data/skimba_data/VAE_Encoder_Features_Semantic20 \
  --dataset-root /data/skimba_data/data_odometry_voxels_all \
  --sequences 08 \
  --num-samples 20 \
  --skimba-bin \
  --latent-shape 8,64,64,8 \
  --device cuda:0 \
  --out-json results/latent_validation.json \
  --out-csv results/latent_validation.csv
```

验证单个 latent/label 对：

```bash
python scripts/data/validate_semantic_vae_latents.py \
  --vae-checkpoint /data/checkpoints/VAE/best_81_x.pth \
  --latent-path /data/latents/000000.bin \
  --label-path /data/labels/000000.label \
  --skimba-bin
```

可用 `--min-miou`、`--min-occ-iou` 和 `--max-ce` 设置质量门槛；不满足时脚本以非零状态退出，适合自动化检查。

## 7. 生成零 partial-condition 特征

入口：`scripts/data/create_zero_partial_condition_features.py`

先预演：

```bash
python scripts/data/create_zero_partial_condition_features.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --output-root /data/Condition_Features_2 \
  --sequences 08 \
  --num-samples 2 \
  --dry-run
```

正式生成：

```bash
python scripts/data/create_zero_partial_condition_features.py \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --output-root /data/Condition_Features_2 \
  --sequences 00,01,02,03,04,05,06,07,08,09,10 \
  --channels 8 \
  --spatial-shape 64,64,8 \
  --out-json results/zero_partial.json
```

默认会覆盖同名输出；若要保留已有文件，加 `--no-overwrite`。

## 8. 检查 image/partial-condition 特征

入口：`scripts/data/check_image_condition_features.py`

检查图像条件：

```bash
python scripts/data/check_image_condition_features.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --condition-root image \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --image-condition-root /data/Image_transform_Voxel_Condition_Features \
  --sequences 08 \
  --channel-candidates 64 \
  --expected-channels 64 \
  --out-json results/image_condition_check.json \
  --out-csv results/image_condition_check.csv
```

检查 partial condition 时将 `--condition-root image` 改为 `partial`，并通过 `--image-condition-root` 传入 partial condition 根目录。当前默认 dataloader/model 读取 raw Mask2Former condition，并取前 24 个语义/辅助通道参与训练；如果文件仍是 64 通道中间产物，检查时使用 `--channel-candidates 24,64 --expected-channels 64`。检查失败时进程返回状态码 1。

## 9. 生成 Mask2Former 语义 partial-condition 特征

该流程用 Mask2Former 导出的 2D semantic probability map 替换原 partial-condition 分支。Mask2Former 负责语义识别，MonoScene `vox2pix/FLoSP` 只负责把固定通道的 2D 语义概率搬运到 latent voxel grid。

推荐先使用 Cityscapes semantic segmentation R50 checkpoint。Cityscapes 是街景数据，比 ADE20K 更接近 SemanticKITTI 图像；当前 `config/mask2former_to_semantickitti.yaml` 也按 Cityscapes 19 个 semantic train id 顺序映射到 SemanticKITTI learning labels。

下载权重：

```bash
cd /path/to/project-master/core-code/Mask2Former-main
mkdir -p weights
wget -O weights/cityscapes_semantic_R50.pkl \
  https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl
```

输入语义文件默认路径：

```text
<semantic-root>/sequences/<seq>/image_2/<frame>.npz
```

每个 `.npz` 必须包含：

```text
probabilities: float32 [num_model_classes,H,W]
```

先导出 Mask2Former 2D 概率。预演命令只检查输入/输出路径，不加载 Detectron2 或 checkpoint：

```bash
python scripts/data/export_mask2former_semantic_probabilities.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --kitti-root /data/SemanticKITTI \
  --output-root /data/Mask2Former_Semantic_Probabilities \
  --mask2former-root ../Mask2Former-main \
  --sequences 08 \
  --num-samples 2 \
  --dry-run \
  --out-json results/mask2former_semantic_probs_dry_run.json \
  --out-csv results/mask2former_semantic_probs_dry_run.csv
```

确认路径正常后去掉 `--dry-run`，并指定 GPU：

```bash
python scripts/data/export_mask2former_semantic_probabilities.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --kitti-root /data/SemanticKITTI \
  --output-root /data/Mask2Former_Semantic_Probabilities \
  --mask2former-root ../Mask2Former-main \
  --sequences 08 \
  --num-samples 2 \
  --overwrite \
  --device cuda:0 \
  --out-json results/mask2former_semantic_probs_sample.json \
  --out-csv results/mask2former_semantic_probs_sample.csv
```

默认使用：

```text
../Mask2Former-main/configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml
../Mask2Former-main/weights/cityscapes_semantic_R50.pkl
```

再把 2D 概率投影成 partial-condition voxel 特征：

```bash
python scripts/data/export_mask2former_partial_condition_features.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --kitti-root /data/SemanticKITTI \
  --semantic-root /data/Mask2Former_Semantic_Probabilities \
  --mapping-path config/mask2former_to_semantickitti.yaml \
  --output-root /data/Mask2Former_Partial_Condition_Features \
  --monoscene-root /opt/MonoScene-master \
  --sequences 08 \
  --num-samples 2 \
  --dry-run \
  --device cuda:0
```

确认路径和 shape 正常后去掉 `--dry-run` 并生成全训练序列。最终输出为：

```text
<output-root>/sequences/<seq>/voxels/<frame>.bin
```

每个文件是 float32 `[64,64,64,8]`，这是压缩前的中间产物。通道布局：

- `0-19`：映射到 SemanticKITTI learning label 的 20 类概率；无法映射的类别累积到 `0`。
- `20`：最大类别置信度。
- `21`：归一化 entropy，数值越大表示越不确定。
- `22`：2D semantic boundary。
- `23`：有效 2D semantic projection mask。
- `24-63`：预留，第一版填 0。

旧版 8 通道 partial-condition ablation 可把 64 通道语义 condition 离线压缩成训练使用的 8 通道文件：

```bash
python scripts/data/compress_condition_features_to_8.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --input-root /data/Mask2Former_Partial_Condition_Features \
  --output-root /data/Mask2Former_Partial_Condition_Features_8ch \
  --sequences 00,01,02,03,04,05,06,07,08,09,10 \
  --uncertainty-mode entropy \
  --out-json results/mask2former_partial_8ch.json \
  --out-csv results/mask2former_partial_8ch.csv
```

8 通道分组为：`0=road+parking+other-ground`，`1=sidewalk`，`2=car+truck+other-vehicle`，`3=person+bicyclist+motorcyclist+bicycle+motorcycle`，`4=building+fence+pole+traffic-sign`，`5=vegetation+trunk+terrain`，`6=entropy`（或 `--uncertainty-mode low_confidence` 使用 `1-confidence`），`7=valid/FOV`。当前默认训练不走这一步，而是从 raw Mask2Former condition 中取前 24 个通道。

当前默认训练时将 YAML 指向 raw 根目录：

```yaml
dataset_params:
  partial_condition_root: "Mask2Former_Partial_Condition_Features"
```

检查命令：

```bash
python scripts/data/check_image_condition_features.py \
  --config-path config/semantickitti_autoencoder.yaml \
  --condition-root partial \
  --vae-root /data/VAE_Encoder_Features_Semantic20 \
  --image-condition-root /data/Mask2Former_Partial_Condition_Features \
  --sequences 08 \
  --channel-candidates 24,64 \
  --expected-channels 64 \
  --min-std 0.001 \
  --out-json results/mask2former_partial_check.json \
  --out-csv results/mask2former_partial_check.csv
```

注意：该 partial condition 是相机视角语义先验，不是精确 3D 占用。同一条相机射线上的多个 voxel 可能获得同一个 2D 语义概率，几何/遮挡可靠性需要由现有 image condition、VAE latent 和扩散训练共同学习。

## 10. 绘制训练曲线

入口：`plot_training_curves.py`

```bash
python plot_training_curves.py \
  --model-dir /data/model_save_path/Train_LSeg \
  --out-dir /data/model_save_path/Train_LSeg/curves
```

训练脚本把 TensorBoard event 写到 `model_save_path/logs`；绘图脚本默认读取 `model-dir/logs`。脚本读取 `output_*.txt`、`best_*.pth` 和 TensorBoard event 文件，输出 `training_curves.png` 及 CSV。默认在输出目录下新建 `run_###`；若希望直接写入 `--out-dir`，加 `--flat-out-dir`。没有 matplotlib 时仍会生成 CSV。

## 11. 可视化标签修正结果

入口：`dataloader with label rectification/visualize_voxel_label.py`

该脚本没有命令行参数，运行前需要编辑源码中的两个硬编码位置：

```python
dataset_root = "./labels"
with open("./config/semantic-kitti.yaml", "r") as stream:
```

准备 `./labels/*.pt`；每个 `.pt` 需要包含 `voxel_label_org` 和 `voxel_label_rect`。然后运行：

```bash
python "dataloader with label rectification/visualize_voxel_label.py"
```

Open3D 会为每个文件依次打开原标签和修正标签窗口。服务器无图形桌面或 X11 转发时不能正常显示。

## 12. 常见问题

- `ModuleNotFoundError`：先激活正确的 Conda 环境；训练脚本会在参数解析前导入依赖。
- `No ... files found`：确认根目录下是否存在 `sequences/XX/voxels/`，并核对扩展名是 `.bin`、`.label` 还是 `.pt`。
- CUDA OOM：减小 YAML 中的 batch size，或先用 `--num-samples` 验证小批数据。
- checkpoint 加载形状不匹配：checkpoint 的类别数、通道数和模型配置必须一致；不要用命令行参数掩盖真实架构。
- 相对路径异常：始终从 `skimba-main-7-1` 根目录运行命令。
- 只想确认语法：对支持 argparse 的脚本运行 `python <script> --help`；若帮助命令也报依赖错误，先安装对应依赖。
