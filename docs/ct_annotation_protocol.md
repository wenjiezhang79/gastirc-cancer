# CT 标注规范（草稿，供与影像科讨论）

版本：2026-10-31 草稿。最终版本由研究团队与影像科确认后更新。

## 0. 开始标注之前

1. 每个中心先试下载 20–30 例，运行 `bash scripts/02_dicom_summary.sh`，确认静脉期是否齐全、层厚分布（预计以 5 mm 为主）、是否有薄层软组织重建。
2. 确定分析层厚（`ct.target_spacing`）后，全量下载并运行 `bash scripts/03_ct_convert.sh`。
3. **医生只在 `work/ct/native/<study_id>.nii.gz` 上标注**（如在薄层上勾画，用 `work/ct/native_thin/<study_id>.nii.gz`，需开启 `ct.save_thin_series`）。不要在其他工具转换出来的图像上标注，否则坐标可能对不上或被镜像。
4. 汇总结果出来之前不要开始标注。

## 1. 肿瘤中心点（全部病例）

**位置：**
- 在静脉期轴位图上找到肿瘤**最大截面**所在的层面，点在该截面肿瘤的中心。
- 胃壁增厚型（含 Borrmann IV 型、皮革胃）：点在**胃壁最厚处**的壁内中心，不要点在胃腔里。
- 可以借助冠状位、矢状位确认肿瘤范围，但点最终落在肿瘤实体内。

**工具与保存（二选一，不要手抄坐标）：**
- **3D Slicer：** 用 Markups → Point List 放 **1 个**点，保存为 `<study_id>.mrk.json`。
- **ITK-SNAP：** 用画笔在肿瘤中心涂一小团标签（直径约 3–5 mm，标签值任意非 0），保存分割为 `<study_id>_point.nii.gz`。

标注文件统一放到配置 `ct.points.annotation_dir` 指定的目录，然后运行 `bash scripts/03b_import_points.sh`。程序会：
- 把点换算为 native 图像的体素坐标（从 0 开始）；
- 检查点是否在图像范围内、与清单中已有坐标的距离；
- 生成带坐标的清单副本 `private/manifest_with_points.csv`（核对后替换原清单）。

**同时记录（写入清单，每项几秒钟）：**

| 列名 | 取值 | 说明 |
|---|---|---|
| `tumor_visibility` | `clear` / `uncertain` / `invisible` | 肿瘤在 CT 上的可见度；`invisible` 按排除标准处理 |
| `borrmann` | `early` / `I` / `II` / `III` / `IV` / `V` | 早期胃癌填 `early` |

胃充盈程度不需要人工记录，由胃自动分割计算（v2）。

## 2. 肿瘤三维勾画（部分病例）

- **病例选择：** 从**内部训练集**中按 T 分期分层抽取 200–300 例（划分文件见 `work/splits/`），不得使用内部测试集和外部中心病例。
- **层厚：** 在分析层厚（预计 5 mm）的 native 图像上逐层勾画即可；若在薄层上勾画，可隔层勾画后用软件插值，再用 `bash scripts/03c_transfer_masks.sh` 转到分析网格。
- **范围：** 勾画肿瘤实体（含增厚的胃壁），不包括胃腔内容物、周围脂肪和淋巴结。
- **保存：** 二值分割（0 / 1），文件名 `<study_id>_tumor.nii.gz`，在清单中填 `tumor_mask_path`，`mask_source=manual`。

## 3. 两阶段自动分割（v2）

1. TotalSegmentator 自动分割全部病例的胃（无需人工）。程序自动检查中心点到胃的距离（`04_ct_crop.sh` 的 `point_to_stomach_mm`），超过 20 mm 的病例请复核标点。
2. 在胃分割外扩 15–20 mm 的范围内，用人工勾画的病例训练 nnU-Net 分割肿瘤；其余病例自动分割，用中心点挑出肿瘤连通域，医生审核修改后保存，`mask_source=auto_reviewed`。

## 4. 质控

运行 `bash scripts/04_ct_crop.sh` 后：
- 查看运行目录下的 `overview_*_pNN.png` 总览图（每格一例：青色十字为中心点，红色为肿瘤分割），抽查是否标偏；
- `status.csv` 中 `point_to_tumor_mm`、`point_to_stomach_mm`、`pad_frac` 异常的病例逐例复核；
- 质控图是患者影像，只在服务器上查看。
