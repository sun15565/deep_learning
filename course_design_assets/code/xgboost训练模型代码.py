import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("害情数据.csv", encoding="utf-8-sig")

# =========================
# 2. 构建高风险标签
# =========================
df["风险分位"] = (
    df.groupby(["害情名称", "采集方法"])["害情数值"]
      .rank(pct=True, method="average")
)

df["高风险"] = (df["风险分位"] >= 0.75).astype(int)

# =========================
# 3. 构建历史情情特征
# =========================
df = df.sort_values(
    ["地点", "害情名称", "采集方法", "观测年份", "标准周"]
).reset_index(drop=True)

group_cols = ["地点", "害情名称", "采集方法"]
g = df.groupby(group_cols, sort=False)["害情数值"]

# 只使用当前之前的历史值，避免信息泄露
df["lag_1"] = g.shift(1)
df["lag_2"] = g.shift(2)
df["lag_3"] = g.shift(3)

past = g.shift(1)

df["roll_mean_3"] = (
    past.groupby([df[c] for c in group_cols])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
)

df["roll_mean_5"] = (
    past.groupby([df[c] for c in group_cols])
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
)

df["hist_mean"] = (
    past.groupby([df[c] for c in group_cols])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
)

df["hist_max"] = (
    past.groupby([df[c] for c in group_cols])
        .expanding(min_periods=1)
        .max()
        .reset_index(level=[0, 1, 2], drop=True)
)

# 同标准周历史均值
group_week_cols = ["地点", "害情名称", "采集方法", "标准周"]

past_week = (
    df.groupby(group_week_cols, sort=False)["害情数值"]
      .shift(1)
)

df["same_week_hist_mean"] = (
    past_week.groupby([df[c] for c in group_week_cols])
             .expanding(min_periods=1)
             .mean()
             .reset_index(level=[0, 1, 2, 3], drop=True)
)

# 近期偏离历史均值
df["recent_minus_hist"] = df["lag_1"] - df["hist_mean"]

history_features = [
    "lag_1", "lag_2", "lag_3",
    "roll_mean_3", "roll_mean_5",
    "hist_mean", "hist_max",
    "same_week_hist_mean",
    "recent_minus_hist"
]

df["has_history"] = df["lag_1"].notna().astype(int)
df[history_features] = df[history_features].fillna(0)

# =========================
# 4. 时间外推验证
# =========================
split_year = int(df["观测年份"].quantile(0.8))

train_df = df[df["观测年份"] <= split_year].copy()
test_df = df[df["观测年份"] > split_year].copy()

num_features = [
    "标准周",
    "lag_1", "lag_2", "lag_3",
    "roll_mean_3", "roll_mean_5",
    "hist_mean", "hist_max",
    "same_week_hist_mean",
    "recent_minus_hist",
    "has_history"
]

cat_features = ["害情名称", "采集方法", "地点"]

X_all = pd.get_dummies(
    df[num_features + cat_features],
    columns=cat_features
)

X_train = X_all.loc[train_df.index]
X_test = X_all.loc[test_df.index]

y_train = train_df["高风险"]
y_test = test_df["高风险"]

# 类别不平衡权重
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# =========================
# 5. XGBoost 模型
# =========================
model = XGBClassifier(
    n_estimators=450,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=4,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# 6. 预测与评价
# =========================
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "AUC": roc_auc_score(y_test, prob),
    "Precision": precision_score(y_test, pred, zero_division=0),
    "Recall": recall_score(y_test, pred, zero_division=0),
    "F1-score": f1_score(y_test, pred, zero_division=0)
}

cm = confusion_matrix(y_test, pred)

print("时间切分年份：", split_year)
print("训练集样本数：", len(train_df))
print("测试集样本数：", len(test_df))

print("\nXGBoost模型评价指标：")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\n混淆矩阵：")
print(cm)

# =========================
# 7. 特征重要性
# =========================
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

agg_importance = {}

for f in num_features:
    agg_importance[f] = importance.get(f, 0)

for cat in cat_features:
    cols = [c for c in importance.index if c.startswith(cat + "_")]
    agg_importance[cat] = importance[cols].sum()

agg_importance = (
    pd.Series(agg_importance)
      .sort_values(ascending=False)
)

print("\n聚合后的特征重要性：")
print(agg_importance)

# =========================
# 8. 保存结果
# =========================
pd.DataFrame([metrics]).to_csv(
    "xgboost_historical_model_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(
    cm,
    index=["真实非高风险", "真实高风险"],
    columns=["预测非高风险", "预测高风险"]
).to_csv(
    "xgboost_historical_model_confusion_matrix.csv",
    encoding="utf-8-sig"
)

agg_importance.to_csv(
    "xgboost_historical_model_feature_importance.csv",
    encoding="utf-8-sig"
)

# 图1：聚合特征重要性
plt.figure(figsize=(8, 5))
agg_importance.sort_values().plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("XGBoost 历史情情风险模型特征重要性")
plt.tight_layout()
plt.savefig("xgboost_historical_model_feature_importance.png", dpi=300)
agg_importance.reset_index().rename(
    columns={"index": "feature", 0: "importance"}
).to_csv(
    "plot_data_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

# 图2：混淆矩阵热力图
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set_xticks([0, 1], labels=["预测非高风险", "预测高风险"])
ax.set_yticks([0, 1], labels=["真实非高风险", "真实高风险"])
ax.set_xlabel("预测标签")
ax.set_ylabel("真实标签")
ax.set_title("混淆矩阵热力图")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("xgboost_historical_model_confusion_heatmap.png", dpi=300)
pd.DataFrame(
    cm,
    index=["真实非高风险", "真实高风险"],
    columns=["预测非高风险", "预测高风险"]
).to_csv(
    "plot_data_confusion_matrix_heatmap.csv",
    encoding="utf-8-sig"
)

# 图3：ROC 曲线
fpr, tpr, _ = roc_curve(y_test, prob)
roc_auc = roc_auc_score(y_test, prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="随机分类器")
plt.xlabel("假阳性率(FPR)")
plt.ylabel("真正率(TPR)")
plt.title("ROC 曲线")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("xgboost_historical_model_roc_curve.png", dpi=300)
pd.DataFrame(
    {"fpr": fpr, "tpr": tpr}
).to_csv(
    "plot_data_roc_curve.csv",
    index=False,
    encoding="utf-8-sig"
)

# 图4：PR 曲线
precision, recall, _ = precision_recall_curve(y_test, prob)
ap = average_precision_score(y_test, prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
plt.xlabel("召回率(Recall)")
plt.ylabel("精确率(Precision)")
plt.title("Precision-Recall 曲线")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("xgboost_historical_model_pr_curve.png", dpi=300)
pd.DataFrame(
    {"recall": recall, "precision": precision}
).to_csv(
    "plot_data_pr_curve.csv",
    index=False,
    encoding="utf-8-sig"
)

# 图5：预测概率分布
plt.figure(figsize=(7, 5))
plt.hist(prob[y_test == 0], bins=30, alpha=0.7, label="真实非高风险", density=True)
plt.hist(prob[y_test == 1], bins=30, alpha=0.7, label="真实高风险", density=True)
plt.xlabel("预测为高风险的概率")
plt.ylabel("密度")
plt.title("测试集预测概率分布")
plt.legend()
plt.tight_layout()
plt.savefig("xgboost_historical_model_probability_distribution.png", dpi=300)
bins = np.linspace(0, 1, 31)
hist_neg, bin_edges = np.histogram(prob[y_test == 0], bins=bins, density=True)
hist_pos, _ = np.histogram(prob[y_test == 1], bins=bins, density=True)
pd.DataFrame(
    {
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "density_true_non_high_risk": hist_neg,
        "density_true_high_risk": hist_pos
    }
).to_csv(
    "plot_data_probability_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.close("all")
