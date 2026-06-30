# deep_learning

本仓库用于存放深度学习课程作业与课程设计材料。课程设计主题为“面向农作物病虫害识别与无人机防治决策的深度学习应用”，主要内容包括 YOLO11s 基线模型、改进模型实验、病虫害识别平台、害情-气象风险预测以及无人机变量施药方案。

## 目录结构

```text
.
├── README.md
├── HW01-20234080114-刘佳雨.ipynb
├── HW02-20234080114-刘佳雨.ipynb
├── HW02-20234080114-刘佳雨-请看这个.ipynb
├── HW03-20234080114-刘佳雨.ipynb
├── HW04-20234080114-刘佳雨.ipynb
├── 课程设计-刘佳雨-20234080114.ipynb
└── course_design_assets/
    ├── code/
    ├── figures/
    ├── model result/
    ├── platform/
    ├── resultfile/
    ├── uav_farmland_output_square/
    ├── DL课程设计_修改前备份.ipynb
    └── 害情数据.csv
```

## 根目录文件说明

| 文件 | 说明 |
| --- | --- |
| `HW01-20234080114-刘佳雨.ipynb` | 深度学习课程第 1 次作业 |
| `HW02-20234080114-刘佳雨.ipynb` | 深度学习课程第 2 次作业 |
| `HW02-20234080114-刘佳雨-请看这个.ipynb` | 第 2 次作业的主要查看版本 |
| `HW03-20234080114-刘佳雨.ipynb` | 深度学习课程第 3 次作业 |
| `HW04-20234080114-刘佳雨.ipynb` | 深度学习课程第 4 次作业 |
| `课程设计-刘佳雨-20234080114.ipynb` | 课程设计主报告与实验展示 notebook |

## 课程设计资源目录

`course_design_assets/` 用于统一存放课程设计 notebook 引用的代码、图片、模型结果和平台文件，避免 GitHub 预览时图片路径失效。

| 路径 | 内容 |
| --- | --- |
| `course_design_assets/code/` | 模型训练、农田建模、无人机路径规划、风险预测等实验代码 |
| `course_design_assets/figures/` | 报告插图、模型结构图、训练曲线、平台截图和可视化结果 |
| `course_design_assets/model result/` | YOLO11s Baseline、DBRA、GatedAttn、MSAFE、Improved 等模型训练输出 |
| `course_design_assets/platform/` | 病虫害识别 Web 平台代码、模板、上传图片、识别结果和模型文件 |
| `course_design_assets/resultfile/` | 无人机调度、喷洒路径、资源消耗、风险点等中间结果 CSV |
| `course_design_assets/uav_farmland_output_square/` | 农田空间建模与无人机路径规划相关输出图 |
| `course_design_assets/害情数据.csv` | 害情-气象风险预测实验使用的数据文件 |

## 查看说明

建议优先打开根目录下的 `课程设计-刘佳雨-20234080114.ipynb`。该 notebook 中的图片路径已经调整为 `course_design_assets/...`，可直接在 GitHub 页面中预览。

如果在本地运行 notebook，请保持 `课程设计-刘佳雨-20234080114.ipynb` 与 `course_design_assets/` 位于同一目录层级，否则部分图片、CSV 或模型输出文件可能无法读取。
