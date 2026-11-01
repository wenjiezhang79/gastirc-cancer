# 胃癌多模态多任务预测模型（gastric_mm）

用治疗前的胃镜活检 HE 切片和静脉期增强 CT，同时预测 Lauren 分型、6 项生物标志物和预后。

- 项目规则：`CLAUDE.md`
- 课题设计：`docs/design.md`
- 方案决策汇总：`docs/decisions.md`
- CT 标注规范（草稿）：`docs/ct_annotation_protocol.md`

当前版本：**v0 数据层** + **v1 病理多任务模型**（代码完成，尚未在真实数据上运行）。

---

## 一、在服务器上第一次运行

```bash
# 1. 环境检查（不联网、不读患者数据），把生成的 env_report.md 贴回
bash scripts/00_env_check.sh

# 2. 用假数据把全流程跑一遍，确认环境没问题（约 1 分钟，CPU 即可）
bash scripts/run_v0_demo.sh

# 3. 准备自己的配置：复制模板后修改路径
cp configs/server_example.yaml configs/server.yaml
```

之后每个步骤都是 `bash scripts/<脚本>.sh`，默认读取 `configs/server.yaml`。如需指定其他配置或 Python：

```bash
CONFIG=configs/其他.yaml PYTHON=/path/to/env/bin/python bash scripts/03_ct_convert.sh --limit 5
```

**建议每一步先用 `--limit 5` 跑 3–5 例，确认没问题再跑全量。**

## 二、流程与脚本

| 步骤 | 脚本 | 作用 | 主要输出 |
|---|---|---|---|
| 0 | `00_env_check.sh` | 环境检查 | `env_report.md`（贴回） |
| 1 | `01_validate_manifest.sh` | 清单校验、标签派生预览 | `validation_report.md`（可贴回） |
| 1b | `01b_split.sh` | 内部中心 8:2 + 训练集 5 折（迭代分层） | `work/splits/split_seed2026.csv`、`split_report.md` |
| 2 | `02_dicom_summary.sh` | 汇总期相、层厚、厂家、卷积核等 | `dicom_summary.md`（可贴回）、`dicom_series.csv` |
| 3 | `03_ct_convert.sh` | DICOM → NIfTI（原始分辨率，医生在此标注）+ 整卷重采样 | `work/ct/native/`、`work/ct/resampled/` |
| 3b | `03b_import_points.sh` | 导入医生在 Slicer / ITK-SNAP 中标的中心点 | `points_report.csv`、`private/manifest_with_points.csv` |
| 3c | `03c_transfer_masks.sh` | 薄层上勾画的分割转换到分析网格 | 分割文件、`private/manifest_with_masks.csv` |
| 4 | `04_ct_crop.sh` | 以中心点为中心裁剪 ROI、窗宽标准化、质控图 | `work/ct/crops/`、`work/ct/qc/`、总览图 |
| 5 | `05_wsi_tiling.sh` | 组织检测、20× 切块（只存坐标）、质控图 | `work/wsi/tiles/`、`work/wsi/qc/` |
| 5b | `05b_import_trident.sh` | 导入 TRIDENT 已提取的 7 个编码器特征 | `work/wsi/features/<编码器>/`、`trident_import_report.md`（可贴回） |
| 6 | `06_wsi_features.sh` | 用本项目流程提取特征（新增病例 / 染色增强时用） | `work/wsi/features/<编码器>/` |

每次运行都会在 `outputs/<步骤>/<时间戳>/` 下保存 `config.yaml`（实际使用的配置）、`run.log`、`env.json`、`status.csv`（每例状态）和报告。已完成的病例默认跳过，加 `--overwrite` 重做；只重跑部分病例用 `--study-ids C1-0001,C1-0002`。

退出码：0 = 全部成功；1 = 配置或数据需要处理（看提示）；2 = 程序内部错误（把 run.log 末尾贴回）；3 = 部分病例失败（见 status.csv）。

## 二b、v1 病理多任务模型

前提：已完成清单校验（01）、数据划分（01b）、TRIDENT 特征导入（05b）。v1 的参数都在配置的 `v1:` 一节。

**1. 编码器初筛**（7 个编码器，只训练 Lauren + MSI，每个 3 个随机种子，按预先规定的指标排名）：

```bash
GPU=0 ENCODERS="uni2_h virchow2 conch_v15 musk" bash scripts/11_v1_compare_encoders.sh
```

```bash
GPU=1 ENCODERS="gigapath chief_ctranspath resnet50_trunc" bash scripts/11_v1_compare_encoders.sh
```

两张卡都跑完后，汇总比较：

```bash
COMPARE_ONLY=1 bash scripts/11_v1_compare_encoders.sh
```

**2. 完整多任务模型**（选定编码器后；可以两张卡分折并行，同一个实验名）：

```bash
EXP=uni2h_abmil GPU=0 bash scripts/10_v1_train.sh --encoders uni2_h --folds 0,1,2
```

```bash
EXP=uni2h_abmil GPU=1 bash scripts/10_v1_train.sh --encoders uni2_h --folds 3,4
```

**3. 消融实验**（单任务 vs 多任务、TransMIL、均值池化、固定权重、Cox）：

```bash
ENCODER=uni2_h GPU=0 bash scripts/13_v1_ablation.sh
```

**4. 最终评估**（模型定型后只做一次；不加 `--final` 不会运行，同一模型对同一队列重复运行会被拒绝）：

```bash
EXP=uni2h_abmil COHORT=test bash scripts/12_v1_predict.sh --final
```

| 输出（`outputs/path_mtl/<实验名>/`） | 内容 |
|---|---|
| `cv_report.md` | 交叉验证报告：各任务折外指标与 95% CI、各折均值 ± 标准差（可贴回） |
| `oof_predictions.csv` | 每例折外预测（概率、风险评分、1/3/5 年生存概率）（不外传） |
| `postprocess.json` | 由折外预测确定的阈值与风险分组截断值（之后冻结） |
| `oof_attention.h5`、`oof_embeddings.h5` | 注意力（热图用）与病理 token（融合用） |
| `seed*/fold*/` | 每个模型、训练曲线、病例分配 |
| `_compare/<时间>/compare_report.md` | 编码器比较表（可贴回） |
| `final_<队列>_<时间>/report.md` | 最终评估报告（可贴回） |

先用假特征验证环境（CPU，约 2 分钟）：`bash scripts/run_v1_demo.sh`。

## 三、哪些可以贴回、哪些不能

| 可以贴回（已确认只含 study_id） | **不能拷出服务器** |
|---|---|
| `env_report.md`、`validation_report.md`、`dicom_summary.md`、`split_report.md`、`trident_import_report.md`、`cv_report.md`、`compare_report.md`、最终评估 `report.md`、`status.csv`、`run.log` | 任何 `private/` 目录（含原始文件名，可能带住院号） |
| | 质控图、总览图（是患者影像） |
| | 清单本身、`labels_preview.csv`、划分文件、`oof_predictions.csv`、`predictions.csv`（逐例结果） |

贴回日志前请再扫一眼，确认没有姓名或住院号。程序在设计上不会把病人级路径写进日志，但第三方库的报错偶尔可能带出。

## 四、离线安装

服务器上已经能运行 TRIDENT 和各病理基础模型，**不要整体重装，以免改动已能运行的 torch / timm 版本**。先运行 `00_env_check.sh`，只补装报告中标为"未安装"的必需包（通常是 `pydicom`、`SimpleITK` 等）。

在一台**操作系统和 Python 版本与服务器相同**、可以联网的机器上：

```bash
pip download -d wheels --only-binary=:all: pydicom==2.4.4 SimpleITK==2.3.1
# 或下载全部：pip download -d wheels --only-binary=:all: -r requirements.txt
```

如果联网机器的系统或 Python 版本不同，可以指定目标平台（`X.Y` 换成服务器的 Python 版本，如 3.10）：

```bash
pip download -d wheels --only-binary=:all: --python-version X.Y --platform manylinux2014_x86_64 -r requirements.txt
```

把 `wheels/` 拷到服务器后：

```bash
pip install --no-index --find-links wheels pydicom SimpleITK
```

所有依赖都是预编译 wheel，不需要编译器。`openslide-bin` 自带 OpenSlide 动态库，无需系统安装。

## 五、数据约定（摘要）

- 病例清单列名见 `CLAUDE.md` 5.1；模板：`configs/manifest_template.csv`。可选扩展列：`<任务>_source`（单项标签来源）、`tumor_visibility`、`borrmann`、`ct_series_number`（强制指定 CT 序列号）。
- 清单中只用研究编号 `study_id`（如 `C1-0001`），不得出现姓名、住院号列。
- `ct_point_x/y/z` 是 `work/ct/native/<study_id>.nii.gz` 的体素坐标，**从 0 开始**；请用 `03b_import_points.sh` 从标注文件导入，不要手抄。
- 路径可以是绝对路径，也可以是相对于 `paths.root` 的相对路径。

## 六、开发者

```bash
pip install -r requirements-dev.txt
python -m pytest            # 全部测试在 CPU 上用假数据运行，约 1 分钟，测试中禁止联网
```

代码结构：

```
gastric_mm/
├── config.py  paths.py  manifest.py  labels.py  split.py  fake_data.py  env_check.py  metrics.py  losses.py
├── models/   mil.py  heads.py  path_model.py
├── training/ bag_dataset.py  survival_bins.py  task_spec.py  trainer.py
├── path_mtl/ train.py  predict.py  compare.py  evaluate.py  common.py
├── ct/   dicom_io.py  summary.py  convert.py  transforms.py  import_points.py  transfer.py  crop.py  qc.py
├── wsi/  reader.py  tissue.py  tiling.py  extractors.py  extract_features.py  features_io.py  import_trident.py
└── utils/ cli.py  errors.py  log.py  privacy.py  runio.py  runner.py
```
