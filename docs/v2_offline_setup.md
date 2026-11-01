# v2 CT 分支：外部工具与权重的离线准备

服务器上不能直接联网下载。以下内容都在一台**操作系统与 Python 版本和服务器相同**、可以联网的电脑上准备好，
再整体拷到服务器。程序运行时**不会联网**：缺权重时直接报错，告诉你缺什么、该填哪个配置项。

## 0. 总览：哪些装在哪里

| 用途 | 环境 | 需要的包 | 需要的权重 | 配置项 |
|---|---|---|---|---|
| 主流程（特征提取、训练、评估） | 主环境（v0/v1 已在用） | 不新增必需包；CT-FM / SuPreM 需要 `monai`、`einops`；Merlin 需要 `merlin-vlm` | 各 CT 编码器权重（见第 4 节） | `v2.features.encoders.<名称>.weights_path` |
| ① 胃分割 | **独立环境** totalseg | `TotalSegmentator`（自带 torch、nnunetv2） | TotalSegmentator 的 total 任务权重 | `v2.external.totalseg.command`、`home_dir` |
| ② 肿瘤分割 | **独立环境** nnunet | `nnunetv2`（自带 torch） | 无（模型在服务器上用我们的勾画训练） | `v2.external.nnunet.bin_dir` |
| ⑤ 影像组学 | **独立环境** radiomics | `pyradiomics`、`pandas`、`scipy`、`PyYAML`、`SimpleITK` | 无 | 运行时用 `RADIOMICS_PY=...` 指定 |

独立环境的原因：这三个工具各自要求特定版本的 torch / numpy，装进主环境可能破坏已经能运行的 TRIDENT 和病理模型。
主流程只通过命令行调用它们，环境之间互不影响。

建议用 conda 或 venv 建环境（下文以 venv 为例，`/opt/envs` 可换成任何目录）。

## 1. TotalSegmentator（胃分割）

**联网电脑上：**

```bash
python -m venv /opt/envs/totalseg
```

```bash
/opt/envs/totalseg/bin/pip download -d wheels_totalseg TotalSegmentator
```

下载权重到一个单独的目录（`TOTALSEG_HOME_DIR` 指定权重目录；`-t total` 为包含胃的基础任务）：

```bash
/opt/envs/totalseg/bin/pip install --no-index --find-links wheels_totalseg TotalSegmentator
```

```bash
TOTALSEG_HOME_DIR=$PWD/totalseg_home /opt/envs/totalseg/bin/totalseg_download_weights -t total
```

**关掉联网的使用统计**：打开 `totalseg_home/config.json`，把 `"send_usage_stats": true` 改成 `false`
（没有这个文件时，先在联网电脑上对任意一例跑一次 TotalSegmentator，它会生成该文件）。

**拷到服务器后：**

```bash
python -m venv /opt/envs/totalseg
```

```bash
/opt/envs/totalseg/bin/pip install --no-index --find-links wheels_totalseg TotalSegmentator
```

在 `configs/server.yaml` 中填写：

```yaml
v2:
  external:
    totalseg:
      command: /opt/envs/totalseg/bin/TotalSegmentator
      home_dir: /data/weights/totalseg_home      # 拷过来的权重目录
```

先跑 1 例确认：`GPU=0 bash scripts/20_stomach_seg.sh --limit 1`。如果报"运行失败"，把 run.log 末尾贴回。

## 2. nnU-Net v2（肿瘤分割）

**联网电脑上：**

```bash
python -m venv /opt/envs/nnunet
```

```bash
/opt/envs/nnunet/bin/pip download -d wheels_nnunet nnunetv2
```

**服务器上：**

```bash
python -m venv /opt/envs/nnunet
```

```bash
/opt/envs/nnunet/bin/pip install --no-index --find-links wheels_nnunet nnunetv2
```

配置 `v2.external.nnunet.bin_dir: /opt/envs/nnunet/bin`。数据集与结果目录默认在 `<work_dir>/ct/nnunet/` 下，
训练脚本 `nnunet_train.sh` 由 `MODE=train bash scripts/22_nnunet_prepare.sh` 生成，环境变量已写在脚本中。

## 3. pyradiomics（影像组学）

pyradiomics **没有预编译 wheel**，需要编译（要先装好 numpy，并有 C 编译器 gcc）。

**方法 A（推荐）：在联网电脑上编译成 wheel，再拷到服务器。** 联网电脑的操作系统、Python 版本必须和服务器相同：

```bash
python -m venv /opt/envs/radiomics
```

```bash
/opt/envs/radiomics/bin/pip install "numpy<2" setuptools wheel versioneer
```

```bash
/opt/envs/radiomics/bin/pip wheel --no-build-isolation -w wheels_radiomics pyradiomics
```

```bash
/opt/envs/radiomics/bin/pip download -d wheels_radiomics "numpy<2" setuptools wheel versioneer pandas scipy PyYAML SimpleITK PyWavelets pykwalify six
```

服务器上：

```bash
python -m venv /opt/envs/radiomics
```

```bash
/opt/envs/radiomics/bin/pip install --no-index --find-links wheels_radiomics pyradiomics pandas scipy PyYAML
```

**方法 B：服务器上有 gcc 时直接在服务器编译**（先离线装好 numpy 等，再 `pip install --no-build-isolation pyradiomics`）。

运行：`RADIOMICS_PY=/opt/envs/radiomics/bin/python bash scripts/25_radiomics.sh --num-workers 8`。
本项目的影像组学模块不依赖 torch，该环境里不需要装 torch。

## 4. CT 编码器权重

| 编码器 | 类型 | 需要的文件 | 来源（以官方说明为准） | 额外的包 |
|---|---|---|---|---|
| `imagenet_r50` | 2.5D | torchvision ResNet50 ImageNet 权重（`resnet50-*.pth`），**可与病理的 resnet50_trunc 共用同一文件** | torchvision | 无 |
| `radimagenet_r50` | 2.5D | RadImageNet ResNet50 的 **PyTorch** 权重 | RadImageNet 官方仓库说明中的下载方式 | 无 |
| `dinov2_vitb14` | 2.5D | `dinov2_vitb14_pretrain.pth`（官方），或 timm 的 `vit_base_patch14_dinov2` 权重 | facebookresearch/dinov2 或 timm | 无（.safetensors 需 safetensors，timm 已依赖） |
| `ctfm` | 3D | CT-FM 特征提取器的 `model.safetensors` | Hugging Face `project-lighter/ct_fm_feature_extractor` | `monai` |
| `suprem` | 3D | `supervised_suprem_swinunetr_2100.pth` | SuPreM 官方仓库 | `monai`、`einops` |
| `merlin` | 3D | Merlin 官方 checkpoint（`.pt`） | Hugging Face `stanfordmimi/Merlin` | `merlin-vlm` |

`monai`、`einops` 是纯 Python 包，在联网电脑上 `pip download -d wheels monai einops --no-deps` 后拷到服务器，
用 `pip install --no-index --find-links wheels monai einops` 安装到主环境（monai 依赖的 numpy、torch 主环境已有）。
`merlin-vlm` 依赖较多（如 transformers），如与主环境冲突，可以单独建一个环境专门跑 Merlin 的特征提取
（特征提取模块可以在任何装有本项目依赖的环境中运行，输出格式统一）。

在 `configs/server.yaml` 中填写：

```yaml
v2:
  features:
    encoders:
      imagenet_r50: {weights_path: /data/weights/resnet50/resnet50-0676ba61.pth}
      radimagenet_r50: {weights_path: /data/weights/radimagenet/ResNet50.pt}
      dinov2_vitb14: {weights_path: /data/weights/dinov2/dinov2_vitb14_pretrain.pth}
      ctfm: {weights_path: /data/weights/ct_fm/model.safetensors}
      suprem: {weights_path: /data/weights/suprem/supervised_suprem_swinunetr_2100.pth}
      merlin: {weights_path: /data/weights/merlin/merlin.pt}
```

`bash scripts/00_env_check.sh --hash` 会列出这些权重是否存在及其 SHA256。

## 5. 每个编码器第一次使用前：与官方代码核对（3–5 例）

RadImageNet 与三个 3D 基础模型的网络结构、权重键名和预处理是按公开资料实现的，**必须核对一次**：

1. 提取 3 例，同时保存送入编码器的输入数组：

   ```bash
   ENCODER=ctfm GPU=0 bash scripts/24_ct_features.sh --limit 3 --dump-inputs 3
   ```

   输入数组与本程序算出的特征保存在运行目录 `private/encoder_inputs/<study_id>_<ROI>.npz`
   （`model_input` 与 `features` 两项；这是患者影像，只在服务器上使用）。
2. 在官方示例代码中加载同一权重，对 `model_input` 做前向，得到官方特征。
3. 比较两者的余弦相似度，应 > 0.99：

   ```python
   import numpy as np
   z = np.load("C1-0001_tumor.npz")
   ours = z["features"]            # 本程序的特征
   theirs = ...                    # 官方代码对 z["model_input"] 的输出
   cos = (ours * theirs).sum(-1) / np.linalg.norm(ours, axis=-1) / np.linalg.norm(theirs, axis=-1)
   print(cos)
   ```

4. 再对照官方说明检查**预处理**是否一致（体素大小、方向、HU 截断与缩放范围）。本程序的设定：

   | 编码器 | ROI 规格 | 方向 | 强度 |
   |---|---|---|---|
   | 2.5D（三个） | tumor_96mm / regional_160mm（0.8 × 0.8 × 5 mm），中心层 ±3 / ±7 层 | 轴位 | 软组织窗 40/400 → [0,1] → ImageNet 均值方差（RadImageNet 可在配置中改 mean/std） |
   | `ctfm` | hu_tumor_96mm / hu_regional_160mm | SPL | HU −1024–2048 → [0,1] |
   | `suprem` | suprem_144mm（1.5 mm 各向同性，96³） | RAS | HU −175–250 → [0,1] |
   | `merlin` | merlin_abdomen（1.5 × 1.5 × 3 mm，224 × 224 × 160） | RAS | HU −1000–1000 → [0,1] |

核对不通过（相似度低或报权重不匹配）时，把报错和核对结果贴回，由开发者调整适配代码；**核对通过前不要全量提取**。

## 6. 用假数据确认整个 v2 流程

```bash
bash scripts/run_v2_demo.sh
```

该演示用假的 TotalSegmentator / nnU-Net 命令（`gastric_mm/fake_tools/`）代替真实工具，只检验本项目的封装与前后处理，
约 3 分钟，CPU 即可。真实工具是否可用，请按第 1–3 节各跑 1 例确认。
