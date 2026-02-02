# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
from scipy import stats

# Advanced Modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from xgboost import XGBRegressor
import shap
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 0. Style Setup
# ==========================================
def set_style():
    try:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    except:
        plt.style.use('seaborn-whitegrid')
        
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    global C_MAIN
    C_MAIN = "#004aad" 

set_style()
warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Pipeline
# ==========================================

class DataIntegrator:
    def __init__(self, t1_path, raw_path):
        self.t1_path = t1_path
        self.raw_path = raw_path
        
    def _find_file(self, filename):
        dirs = [os.getcwd(), r"C:\Users\85269\OneDrive\Desktop\T3V2", 
                r"C:\Users\85269\OneDrive\Desktop"]
        if os.path.exists(filename): return filename
        for d in dirs:
            if not d: continue
            fp = os.path.join(d, filename)
            if os.path.exists(fp): return fp
        raise FileNotFoundError(f"Missing {filename}")

    def load_and_merge(self):
        print("[1] Loading Data...")
        t1_path = self._find_file(self.t1_path)
        raw_path = self._find_file(self.raw_path)
        
        t1_df = pd.read_csv(t1_path)
        t1_df['dancer_clean'] = t1_df['dancer'].str.strip().str.lower()
        
        try: raw_df = pd.read_csv(raw_path, encoding='utf-8-sig')
        except: raw_df = pd.read_csv(raw_path, encoding='ISO-8859-1')
        
        raw_df.columns = raw_df.columns.str.strip().str.replace('ï»¿', '')
        if 'celebrity_name' not in raw_df.columns:
            col = [c for c in raw_df.columns if 'celebrity' in c.lower()][0]
            raw_df.rename(columns={col: 'celebrity_name'}, inplace=True)
            
        # Melt
        score_cols = [c for c in raw_df.columns if 'judge' in c and 'score' in c]
        meta_cols = ['season', 'celebrity_name', 'ballroom_partner', 
                     'celebrity_age_during_season', 'celebrity_gender']
        
        melted = raw_df.melt(id_vars=[c for c in meta_cols if c in raw_df.columns], 
                             value_vars=score_cols, value_name='score', var_name='w')
        melted['week'] = melted['w'].str.extract(r'week(\d+)_').astype(float)
        melted = melted.dropna(subset=['score', 'week'])
        
        # Aggregation
        agg_funcs = {'score': 'sum'}
        for c in meta_cols: 
            if c in melted.columns and c not in ['season', 'celebrity_name']:
                agg_funcs[c] = 'first'
                
        feat_df = melted.groupby(['season', 'week', 'celebrity_name']).agg(agg_funcs).reset_index()
        feat_df.rename(columns={'score': 'judge_total'}, inplace=True)
        feat_df['dancer_clean'] = feat_df['celebrity_name'].str.strip().str.lower()
        
        # Merge
        full = pd.merge(t1_df, feat_df, on=['season', 'week', 'dancer_clean'], how='inner')
        full['judge_share'] = full.groupby(['season', 'week'])['judge_total'].transform(lambda x: x/x.sum())
        
        print(f"    Merged Shape: {full.shape}")
        return full

# ==========================================
# 2. Analysis Modules
# ==========================================

def analyze_kingmaker(df):
    print("\n[Analysis] Kingmaker Index...")
    df['premium'] = df['p_hat'] - df['judge_share']
    stats = df.groupby('ballroom_partner')['premium'].agg(['mean', 'count']).reset_index()
    stats = stats[stats['count'] > 5].sort_values('mean', ascending=False)
    stats.rename(columns={'mean': 'PCI'}, inplace=True)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    top_bot = pd.concat([stats.head(5), stats.tail(5)])
    sns.barplot(data=top_bot, x='PCI', y='ballroom_partner', palette='RdBu_r')
    plt.title("Kingmaker Index (Partner Popularity Contribution)", fontweight='bold')
    plt.axvline(0, color='k')
    plt.tight_layout()
    plt.savefig("Task3_Kingmaker.png")
    return stats

def run_econometrics(df):
    """ Robust Elasticity Analysis: LMM -> OLS Fallback """
    print("\n[Analysis] Econometric Elasticity Model...")
    
    # Prep
    model_df = df.dropna(subset=['judge_share', 'celebrity_age_during_season']).copy()
    # Normalize features to help convergence
    model_df['judge_z'] = (model_df['judge_share'] - model_df['judge_share'].mean()) / model_df['judge_share'].std()
    model_df['age_z'] = (model_df['celebrity_age_during_season'] - model_df['celebrity_age_during_season'].mean()) / model_df['celebrity_age_during_season'].std()
    
    formula = "p_hat ~ judge_z + age_z"
    
    try:
        # Attempt 1: Mixed Linear Model
        print("    Attempting Mixed Linear Model (LMM)...")
        md = smf.mixedlm(formula, model_df, groups=model_df["season"])
        # Use more robust optimizer
        mdf = md.fit(method='lbfgs', maxiter=100) 
        print(mdf.summary())
        coef = mdf.params['judge_z']
        
    except Exception as e:
        print(f"    [Warn] LMM Failed ({str(e)}). Falling back to OLS with Fixed Effects.")
        # Attempt 2: OLS with Season Dummy Variables (Fixed Effects)
        formula_fe = "p_hat ~ judge_z + age_z + C(season)"
        md = smf.ols(formula_fe, data=model_df)
        mdf = md.fit()
        print(mdf.summary())
        coef = mdf.params['judge_z']

    # Interpret (Std Beta)
    print(f"\n>> RESULT: Standardized Judge Elasticity = {coef:.4f}")
    print("   (Impact of 1 SD increase in Judge Share on Fan Vote Share)")
    return mdf

def run_shap_analysis(df, pci_stats):
    print("\n[Analysis] Machine Learning Attribution (XGBoost)...")
    
    # Merge PCI
    df = pd.merge(df, pci_stats[['ballroom_partner', 'PCI']], on='ballroom_partner', how='left').fillna(0)
    
    # Prep Data
    features = ['judge_share', 'celebrity_age_during_season', 'PCI', 'week']
    X = df[features].fillna(0)
    y = df['p_hat']
    
    # Model
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    print(f"    Model R2: {model.score(X, y):.4f}")
    
    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    # Plot 1: Summary
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(shap_values, X, show=False, cmap='coolwarm')
    plt.title("Driver Analysis: Global Feature Importance", fontweight='bold')
    plt.tight_layout()
    plt.savefig("Task3_SHAP_Summary.png")
    
    # Plot 2: Controversy Anatomy (Jerry Rice)
    try:
        jerry = df[df['celebrity_name'].str.contains('Jerry Rice', case=False)]
        if not jerry.empty:
            # Pick the week with highest Vote Premium
            idx = (jerry['p_hat'] - jerry['judge_share']).idxmax()
            row_idx = df.index.get_loc(idx)
            
            plt.figure(figsize=(10, 4), dpi=300)
            shap.plots.waterfall(shap_values[row_idx], show=False)
            plt.title(f"Anatomy of a Controversy: Jerry Rice (Week {df.loc[idx, 'week']:.0f})", fontweight='bold')
            plt.tight_layout()
            plt.savefig("Task3_Controversy.png")
            print("    Saved 'Task3_Controversy.png'")
    except Exception as e:
        print(f"    Skipped Jerry Rice plot: {e}")

# ==========================================
# 3. Main
# ==========================================

def main():
    print("="*60)
    print("Task 3: Robust Driver Analysis")
    print("="*60)
    
    t1_file = "Task1_Voting_Inference_Optimized.csv"
    raw_file = "2026_MCM_Problem_C_Data.csv"
    
    try:
        integrator = DataIntegrator(t1_file, raw_file)
        full_data = integrator.load_and_merge()
    except Exception as e:
        print(e); return
        
    # 1. Partner Effect
    pci_stats = analyze_kingmaker(full_data)
    
    # 2. Elasticity (Auto-Fallback enabled)
    run_econometrics(full_data)
    
    # 3. ML Attribution
    run_shap_analysis(full_data, pci_stats)
    
    print("\n[Done] All Task 3 outputs generated.")

if __name__ == "__main__":
    main()
