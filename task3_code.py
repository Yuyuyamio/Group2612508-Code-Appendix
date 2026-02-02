import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import shap
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# 设定中文与负号显示
def set_chinese_plot_style():
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["figure.dpi"] = 130
    mpl.rcParams["savefig.dpi"] = 260
    mpl.rcParams["figure.figsize"] = (10, 6)

# 读取原始数据
def load_raw_data(data_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_csv_path)
    return df

# 解析week和judge列
def parse_week_judge_columns(df: pd.DataFrame):
    pat = re.compile(r"week(\d+)_judge(\d+)_score")
    week_judge_cols = []
    for c in df.columns:
        m = pat.fullmatch(c)
        if m:
            week = int(m.group(1))
            judge = int(m.group(2))
            week_judge_cols.append((week, judge, c))
    week_judge_cols.sort(key=lambda x: (x[0], x[1]))
    weeks = sorted(list({w for w, _, _ in week_judge_cols}))
    return week_judge_cols, weeks

# 宽表转换为长表
def wide_to_long_week_level(df: pd.DataFrame) -> pd.DataFrame:
    week_judge_cols, weeks = parse_week_judge_columns(df)
    id_cols = [
        "celebrity_name", "ballroom_partner", "celebrity_industry", "celebrity_homestate",
        "celebrity_homecountry/region", "celebrity_age_during_season", "season", "results", "placement"
    ]
    rows = []
    for _, r in df.iterrows():
        base = {k: r.get(k, np.nan) for k in id_cols}
        for w in weeks:
            score_cols = [col for (ww, jj, col) in week_judge_cols if ww == w]
            scores = r[score_cols].astype(float)
            if scores.isna().all():
                continue
            total = np.nansum(scores.values)
            if total <= 0:
                continue
            out = base.copy()
            out["week"] = w
            out["judge_total_score"] = float(total)
            out["judge_count"] = int(np.sum(~scores.isna()))
            out["judge_avg_score"] = float(total / max(1, out["judge_count"]))
            rows.append(out)
    long_df = pd.DataFrame(rows)
    long_df["season"] = long_df["season"].astype(int)
    long_df["week"] = long_df["week"].astype(int)
    long_df["celebrity_age_during_season"] = pd.to_numeric(long_df["celebrity_age_during_season"], errors="coerce")
    return long_df

# 添加赛制特征
def add_institution_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    df["Is_Rank"] = ((df["season"] <= 2) | (df["season"] >= 28)).astype(int)
    df["Age_x_IsRank"] = df["celebrity_age_during_season"] * df["Is_Rank"]
    return df

# 加载或生成task1的soft label（估算选票、置信区间宽度、权重）
def load_or_generate_task1_votes(df_long: pd.DataFrame, task1_path="task1_votes.csv") -> pd.DataFrame:
    df = df_long.copy()
    if os.path.exists(task1_path):
        t1 = pd.read_csv(task1_path)
        t1["season"] = t1["season"].astype(int)
        t1["week"] = t1["week"].astype(int)
        merged = df.merge(t1, on=["celebrity_name", "season", "week"], how="left")
        if merged["est_votes"].isna().any() or merged["ci_width"].isna().any():
            print("[警告] task1_votes.csv 与主数据无法完全匹配，存在缺失。")
            merged = generate_fallback_votes(merged)
        else:
            eps = 1e-6
            merged["weight_w"] = 1.0 / (merged["ci_width"].astype(float) + eps)
        return merged
    else:
        print("[提示] 未找到 task1_votes.csv，将自动生成示例 soft label（仅用于跑通流程/出图）。")
        return generate_fallback_votes(df)

# 生成缺失的投票数据
def generate_fallback_votes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    industry = out["celebrity_industry"].fillna("Unknown").astype(str)
    pop_map = {
        "Actor/Actress": 1.15, "Singer/Rapper": 1.10, "Athlete": 1.05, "Model": 1.05,
        "Reality TV": 1.08, "TV Personality": 1.06, "Comedian": 1.03, "Radio Host": 1.12, "Unknown": 1.00
    }
    ind_pop = industry.map(lambda x: pop_map.get(x, 1.00)).astype(float)
    placement = pd.to_numeric(out["placement"], errors="coerce").fillna(out["placement"].median())
    base_pop = 1.0 / (placement + 1.0)
    base_pop = (base_pop - base_pop.min()) / (base_pop.max() - base_pop.min() + 1e-9) + 0.3
    judge = out["judge_total_score"].astype(float)
    judge_scaled = (judge - judge.min()) / (judge.max() - judge.min() + 1e-9) + 0.5
    rng = np.random.default_rng(20260131)
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=len(out))
    est_votes = 1e6 * ind_pop * judge_scaled * base_pop * noise
    out["est_votes"] = est_votes
    season = out["season"].astype(int)
    week = out["week"].astype(int)
    ci = (0.35 + 0.8 * np.exp(-(season - 1) / 6.0)) * (1.0 + 0.15 / (week + 1.0))
    ci_width = ci * out["est_votes"].astype(float) * 0.25
    out["ci_width"] = ci_width
    eps = 1e-6
    out["weight_w"] = 1.0 / (out["ci_width"].astype(float) + eps)
    return out

# 训练和评估模型
def train_xgb_and_shap(df: pd.DataFrame, target_col: str, sample_weight_col=None, model_name="judge"):
    num_features = ["celebrity_age_during_season", "Is_Rank", "Age_x_IsRank", "Partner_PCI", "PCI_x_IsRank", "week"]
    cat_features = ["celebrity_industry"]
    X = df[num_features + cat_features].copy()
    y = df[target_col].astype(float).values
    pre = ColumnTransformer(
        transformers=[("num", "passthrough", num_features), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="drop"
    )
    xgb = XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.0, reg_lambda=1.0, random_state=20260131, n_jobs=4, objective="reg:squarederror"
    )
    pipe = Pipeline([("pre", pre), ("xgb", xgb)])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.22, random_state=20260131
    )
    if sample_weight_col is not None:
        sw = df[sample_weight_col].astype(float).values
        sw_train = sw[np.isin(df.index.values, idx_train)]
        sw_test = sw[np.isin(df.index.values, idx_test)]
    else:
        sw_train = None
        sw_test = None
    pipe.fit(X_train, y_train, xgb__sample_weight=sw_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n[{model_name.upper()}] 目标={target_col}  测试RMSE={rmse:.4f}  R2={r2:.4f}")
    X_all_trans = pipe.named_steps["pre"].transform(X)
    booster = pipe.named_steps["xgb"]
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_all_trans)
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(["celebrity_industry"]))
    feature_names = num_features + ohe_names
    return pipe, explainer, shap_values, X_all_trans, feature_names

# 主程序执行
def main():
    set_chinese_plot_style()
    DATA_CSV = "2026_MCM_Problem_C_Data.csv"
    if not os.path.exists(DATA_CSV):
        alt = "/mnt/data/2026-01-31-13-P3qHBTxqD12d0E2A7DNE.csv"
        if os.path.exists(alt):
            DATA_CSV = alt
        else:
            raise FileNotFoundError("未找到数据文件 2026_MCM_Problem_C_Data.csv，请检查路径。")
    print(f"读取数据：{DATA_CSV}")
    df_raw = load_raw_data(DATA_CSV)
    df_long = wide_to_long_week_level(df_raw)
    print("\n周级长表 df_long：")
    print(df_long.head())
    df_long = add_institution_features(df_long)
    df_long = load_or_generate_task1_votes(df_long)
    df_long.to_csv("out_task3_clean_long.csv", index=False, encoding="utf-8-sig")
    print("\n已保存：out_task3_clean_long.csv")

if __name__ == "__main__":
    main()
