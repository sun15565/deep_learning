# -*- coding: utf-8 -*-
"""
random_forest_weather_risk_visualization.py

随机森林气象环境风险模型 —— 完整版（含丰富可视化）
在原有建模基础上，新增：
  1. 气象因子相关性热图
  2. 不同风险等级下的气象因子均值对比条形图
  3. 气象因子对高风险的箱线图
  4. 随机森林特征重要性条形图
  5. 混淆矩阵热力图
  6. ROC 曲线
  7. 高风险/非高风险样本的预测概率分布直方图
  8. 前两个最重要特征的二维部分依赖图 (PDP)

所有图表均保存至 ./figures/ 目录，可直接插入论文。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import PartialDependenceDisplay

# ------------------------------
# 0. 全局设置：中文字体与图片保存路径
# ------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. 读取数据
# =========================
DATA_PATH = "害情数据.xlsx"
df = pd.read_excel(DATA_PATH)

rename_map = {
    "相对湿度1(%)": "最高湿度(%)",
    "相对湿度2(%)": "最低湿度(%)"
}
df = df.rename(columns=rename_map)

# =========================
# 2. 构建组内风险标签（与原文一致）
# =========================
df["害情数值"] = pd.to_numeric(df["害情数值"], errors="coerce")

# 注：若需严格按“害情名称-采集方法”分组，可将 groupby(["害情名称"]) 改为 groupby(["害情名称", "采集方法"])
df["风险分位"] = (
    df.groupby(["害情名称"])["害情数值"]
      .rank(pct=True, method="average")
)

df["高风险"] = (df["风险分位"] >= 0.75).astype(int)

# 同时构建三级风险标签，用于后续可视化
def risk_level(pct):
    if pct >= 0.75:
        return "高风险"
    elif pct <= 0.25:
        return "低风险"
    else:
        return "中风险"

df["风险等级"] = df["风险分位"].apply(risk_level)

# =========================
# 3. 气象特征整理
# =========================
weather_features = [
    "最高温度(°C)",
    "最低温度(°C)",
    "最高湿度(%)",
    "最低湿度(%)",
    "降雨量(mm)",
    "风速(km/h)",
    "日照时数(hrs)",
    "蒸发量(mm)"
]

for col in weather_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

data = df.dropna(subset=weather_features + ["高风险", "风险分位"]).copy()

X = data[weather_features]
y = data["高风险"]

# =========================
# 4. 划分训练/测试集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. 随机森林模型训练
# =========================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# =========================
# 6. 模型预测与评价指标
# =========================
y_prob = rf_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0)
}

cm = confusion_matrix(y_test, y_pred)

importance = pd.DataFrame({
    "气象因子": weather_features,
    "重要性": rf_model.feature_importances_
}).sort_values("重要性", ascending=False)

# 打印结果
print("随机森林气象风险模型评价指标：")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("\n混淆矩阵：\n", cm)
print("\n气象因子重要性：\n", importance.to_string(index=False))

# ------------------------------
# 7. 保存原始结果表格
# ------------------------------
pd.DataFrame([metrics]).to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_weather_metrics.csv"),
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(
    cm,
    index=["真实非高风险", "真实高风险"],
    columns=["预测非高风险", "预测高风险"]
).to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_weather_confusion_matrix.csv"),
    encoding="utf-8-sig"
)
importance.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_weather_feature_importance.csv"),
    index=False, encoding="utf-8-sig"
)

# ===============================================
# 以下为新增可视化（均可直接用于论文）
# ===============================================

# ---- 图1：气象因子与风险分位的 Pearson 相关系数热图 ----
corr = data[weather_features + ["风险分位"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5)
plt.title("气象因子与风险分位的 Pearson 相关系数", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()
print("图1 相关性热图已保存。")

# ---- 图2：不同风险等级下各气象因子的均值对比条形图（对应论文表4） ----
mean_comparison = data.groupby("风险等级")[weather_features].mean()
# 按风险等级排序
mean_comparison = mean_comparison.reindex(["低风险", "中风险", "高风险"])

# 为便于绘图，将特征名缩短
short_names = {
    "最高温度(°C)": "最高温度",
    "最低温度(°C)": "最低温度",
    "最高湿度(%)": "最高湿度",
    "最低湿度(%)": "最低湿度",
    "降雨量(mm)": "降雨量",
    "风速(km/h)": "风速",
    "日照时数(hrs)": "日照时长",
    "蒸发量(mm)": "蒸发量"
}
mean_plot = mean_comparison.rename(columns=short_names)

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(mean_plot.columns):
    ax = axes[i]
    mean_plot[col].plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title(col, fontsize=12)
    ax.set_ylabel("均值")
    ax.set_xlabel("风险等级")
    ax.tick_params(axis='x', rotation=0)
plt.suptitle("不同风险等级下主要气象因子均值对比", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "risk_level_means_barplot.png"), dpi=300, bbox_inches="tight")
plt.close()
print("图2 风险等级均值对比图已保存。")

# ---- 图3：气象因子在不同高风险状态下的箱线图 ----
# 绘制四个关键因子：最低温度、最低湿度、最高湿度、风速
key_factors = ["最低温度(°C)", "最低湿度(%)", "最高湿度(%)", "风速(km/h)"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, factor in enumerate(key_factors):
    sns.boxplot(x="高风险", y=factor, hue="高风险", data=data, ax=axes[i],
                palette={0: "#4C72B0", 1: "#C44E52"}, dodge=False, hue_order=[0, 1])
    if axes[i].get_legend() is not None:
        axes[i].get_legend().remove()
    axes[i].set_title(f"{factor} 分布（0=非高风险，1=高风险）")
    axes[i].set_xlabel("高风险状态")
plt.suptitle("关键气象因子在不同高风险状态下的分布", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_key_factors.png"), dpi=300)
plt.close()
print("图3 关键因子箱线图已保存。")

# ---- 图4：随机森林特征重要性条形图 ----
plt.figure(figsize=(10, 6))
importance_sorted = importance.iloc[::-1]  # 水平条形图从下往上重要性递减
colors = sns.color_palette("Blues_d", len(importance_sorted))
bars = plt.barh(importance_sorted["气象因子"], importance_sorted["重要性"], color=colors)
plt.xlabel("重要性", fontsize=12)
plt.title("随机森林气象风险模型特征重要性", fontsize=14)
# 在条右侧标注数值
for bar, val in zip(bars, importance_sorted["重要性"]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_barh.png"), dpi=300)
plt.close()
print("图4 特征重要性条形图已保存。")

# ---- 图5：混淆矩阵热力图 ----
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["预测非高风险", "预测高风险"],
            yticklabels=["真实非高风险", "真实高风险"])
plt.title("随机森林气象风险模型混淆矩阵", fontsize=14)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png"), dpi=300)
plt.close()
print("图5 混淆矩阵热力图已保存。")

# ---- 图6：ROC 曲线 ----
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = metrics["AUC"]
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC 曲线 (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("假正例率 (FPR)", fontsize=12)
plt.ylabel("真正例率 (TPR)", fontsize=12)
plt.title("气象风险模型 ROC 曲线", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
plt.close()
print("图6 ROC曲线已保存。")

# ---- 图7：高风险/非高风险样本的预测概率分布直方图 ----
plt.figure(figsize=(8, 5))
sns.histplot(y_prob[y_test == 0], color="#4C72B0", label="非高风险", kde=True, stat="density", bins=30, alpha=0.6)
sns.histplot(y_prob[y_test == 1], color="#C44E52", label="高风险", kde=True, stat="density", bins=30, alpha=0.6)
plt.axvline(0.5, color="black", linestyle="--", label="判定阈值 (0.5)")
plt.xlabel("预测高风险概率", fontsize=12)
plt.ylabel("密度", fontsize=12)
plt.title("高风险与非高风险样本的预测概率分布", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "probability_distribution.png"), dpi=300)
plt.close()
print("图7 预测概率分布图已保存。")
# ---- 图8：二维部分依赖图（前两个最重要特征） ----
top2_features = importance.iloc[0:2, 0].tolist()
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=[(top2_features[0], top2_features[1])],
    grid_resolution=20, ax=ax, n_jobs=1
)
plt.title(f"二维部分依赖图：{top2_features[0]} 与 {top2_features[1]}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "partial_dependence_2d.png"), dpi=300)
plt.close()
print("图8 二维部分依赖图已保存。")

# ---- SHAP 蜜蜂图 ----

# ---- SHAP 蜜蜂图 ----
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test[:500])
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:, :, 1], X_test[:500], feature_names=weather_features,
                  show=False)
plt.title("SHAP 特征贡献蜜蜂图（高风险类别）", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
plt.close()
print("SHAP 蜜蜂图已保存。")

# ========== 精确率-召回率曲线 ==========
prec, rec, _ = precision_recall_curve(y_test, y_prob)
pr_auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(7, 6))
disp = PrecisionRecallDisplay(precision=prec, recall=rec)
disp.plot(color="#C44E52")
plt.title(f"气象风险模型精确率-召回率曲线 (AUC = {pr_auc:.3f})", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=300)
plt.close()
print("PR 曲线已保存。")


