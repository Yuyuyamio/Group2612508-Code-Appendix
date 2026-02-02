# -*- coding: utf-8 -*-
"""
Task 2: Institutional Comparison & Counterfactual Analysis
FINAL MERGED VERSION: Risk S-Curve (Red/Blue) + Power Dilution (Teal/Green)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import make_interp_spline, PchipInterpolator
from typing import Dict, List, Tuple, Optional
import warnings
import os
import sys
import copy
import re
from dataclasses import dataclass
from enum import Enum

# ==========================================
# 0. Visualization Setup
# ==========================================
def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Clean spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['lines.linewidth'] = 3

# --- Palette Definition ---
# Chart 1: Mechanism Contrast
C_RANK = "#004aad"      # Navy Blue
C_PCT = "#d00000"       # Crimson Red

# Chart 2: Power Dilution (The "Grey-Green" one you missed)
C_B2_MAIN = "#2a9d8f"   # Teal/Green (Persian Green)
C_B2_SHADOW = "#e9c46a" # Sand/Gold highlight

set_style()
warnings.filterwarnings('ignore')

# ==========================================
# 1. Structures
# ==========================================

class MechanismType(Enum):
    PERCENT = "percent"
    RANK = "rank"
    BOTTOM_TWO = "bottom_two"

@dataclass
class VoteEvidence:
    season: int
    week: int
    contestant_ids: List[str]
    p_ac: np.ndarray
    p_min: np.ndarray
    p_max: np.ndarray
    judge_scores: np.ndarray
    elimination_count: int
    
    @property
    def n_contestants(self) -> int:
        return len(self.contestant_ids)

@dataclass
class MechanismResult:
    survival_weeks: Dict[str, int]

# ==========================================
# 2. Robust Data Loading
# ==========================================

class DataLoader:
    def __init__(self, task1_filename, raw_filename):
        self.t1_path = self._find_file(task1_filename)
        self.raw_path = self._find_file(raw_filename)

    def _find_file(self, filename):
        dirs = [os.getcwd(), r"C:\Users\85269\OneDrive\Desktop\T1V3", 
                r"C:\Users\85269\OneDrive\Desktop"]
        for d in dirs:
            if not d: continue
            fp = os.path.join(d, filename)
            if os.path.exists(fp): 
                print(f"[System] Found {filename} at {fp}")
                return fp
        raise FileNotFoundError(f"Could not find {filename}")

    def load_evidence(self):
        print("[Data] Loading datasets...")
        t1 = pd.read_csv(self.t1_path)
        t1['dancer_clean'] = t1['dancer'].str.strip().str.lower()
        
        try: raw = pd.read_csv(self.raw_path, encoding='utf-8-sig')
        except: raw = pd.read_csv(self.raw_path, encoding='ISO-8859-1')
        
        raw.columns = raw.columns.str.strip().str.replace('ï»¿', '')
        if 'celebrity_name' not in raw.columns:
            col = [c for c in raw.columns if 'celebrity' in c.lower()][0]
            raw.rename(columns={col: 'celebrity_name'}, inplace=True)

        # Result mapping
        res_map = raw[['season', 'celebrity_name', 'results']].drop_duplicates()
        res_map['dancer_clean'] = res_map['celebrity_name'].str.strip().str.lower()
        
        # Melt Scores
        score_cols = [c for c in raw.columns if 'judge' in c and 'score' in c]
        melted = raw.melt(id_vars=['season', 'celebrity_name'], value_vars=score_cols, value_name='score', var_name='wk_str')
        melted['week'] = melted['wk_str'].str.extract(r'week(\d+)_').astype(float)
        melted = melted.dropna(subset=['score', 'week'])
        
        s_df = melted.groupby(['season', 'celebrity_name', 'week'])['score'].sum().reset_index()
        s_df['dancer_clean'] = s_df['celebrity_name'].str.strip().str.lower()
        s_df.rename(columns={'score': 'S_it'}, inplace=True)
        
        merged = pd.merge(t1, s_df, on=['season', 'week', 'dancer_clean'], how='inner')
        merged = pd.merge(merged, res_map[['season', 'dancer_clean', 'results']], on=['season', 'dancer_clean'], how='left')
        
        evidence = {}
        for (season, week), group in merged.groupby(['season', 'week']):
            group = group.sort_values('dancer')
            d_t = 0
            for r in group['results']:
                r_str = str(r)
                if 'Eliminated Week' in r_str:
                    try:
                        elim_wk = int(re.search(r'Week (\d+)', r_str).group(1))
                        if elim_wk == week: d_t += 1
                    except: pass
            
            evidence[(season, week)] = VoteEvidence(
                season=season, week=week,
                contestant_ids=group['dancer'].tolist(),
                p_ac=group['p_hat'].values,
                p_min=group['p_min'].values,
                p_max=group['p_max'].values,
                judge_scores=group['S_it'].values,
                elimination_count=d_t
            )
        print(f"[Data] Loaded {len(evidence)} weeks.")
        return evidence

# ==========================================
# 3. Simulation Engine
# ==========================================

class Simulator:
    def __init__(self, evidence):
        self.evidence = evidence
        
    def generate_votes(self, ev, n_mc):
        n = ev.n_contestants
        if n == 0: return np.zeros((n_mc, 0))
        raw = np.random.uniform(0, 1, (n_mc, n))
        width = np.maximum(ev.p_max - ev.p_min, 1e-6)
        votes = ev.p_min + raw * width
        votes = votes / votes.sum(axis=1, keepdims=True)
        return votes

    def run(self, season, mechanism, force_elimination=False, n_mc=50):
        weeks = sorted([w for (s, w) in self.evidence.keys() if s == season])
        if not weeks: return []
        results = []
        for _ in range(n_mc):
            alive = set(self.evidence[(season, weeks[0])].contestant_ids)
            survived_until = {c: 0 for c in alive}
            for w in weeks:
                ev = self.evidence[(season, w)]
                curr_ids = [c for c in ev.contestant_ids if c in alive]
                if len(curr_ids) <= 1: break 
                
                idxs = [ev.contestant_ids.index(c) for c in curr_ids]
                S = ev.judge_scores[idxs]
                P = self.generate_votes(ev, 1)[0][idxs]
                P = P / P.sum()
                
                d_t = ev.elimination_count
                if force_elimination and d_t == 0: d_t = 1
                d_t = min(d_t, len(curr_ids)-1)
                
                if d_t > 0:
                    if mechanism == MechanismType.PERCENT:
                        rank_idx = np.argsort((S/S.sum()) + P)
                        elim_local = rank_idx[:d_t]
                    elif mechanism == MechanismType.RANK:
                        SumR = stats.rankdata(-S, method='min') + stats.rankdata(-P, method='min')
                        # Sort Descending SumR
                        df = pd.DataFrame({'idx': range(len(curr_ids)), 'R': SumR, 'P': P})
                        df = df.sort_values(by=['R', 'P'], ascending=[False, True])
                        elim_local = df['idx'].values[:d_t]

                    for e_idx in elim_local:
                        who = curr_ids[e_idx]
                        alive.remove(who)
                
                for c in alive: survived_until[c] = w
            results.append(MechanismResult(survived_until))
        return results

# ==========================================
# 4. Plotting Functions 
# ==========================================

def plot_risk_curve(factors, risk_pct, risk_rank, contestant):
    """ Chart 1: The Red/Blue S-Curve """
    plt.figure(figsize=(10, 6), dpi=300)
    
    x = factors * 100
    try:
        x_new = np.linspace(x.min(), x.max(), 300)
        spl_p = PchipInterpolator(x[::-1], risk_pct[::-1])(x_new)
        spl_r = PchipInterpolator(x[::-1], risk_rank[::-1])(x_new)
    except:
        x_new, spl_p, spl_r = x, risk_pct, risk_rank

    # Percent (Red)
    plt.plot(x_new, spl_p, color=C_PCT, label='Percentage Mechanism', linewidth=3)
    plt.fill_between(x_new, 0, spl_p, color=C_PCT, alpha=0.15)
    
    # Rank (Blue)
    plt.plot(x_new, spl_r, color=C_RANK, label='Rank Mechanism', linewidth=3, linestyle='--')
    plt.fill_between(x_new, 0, spl_r, color=C_RANK, alpha=0.05)
    
    plt.gca().invert_xaxis()
    plt.xlabel("Fan Popularity Strength (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Probability of Elimination (%)", fontsize=12, fontweight='bold')
    plt.title(f"Mechanism Stress Test: {contestant}", fontsize=14, fontweight='bold', pad=15)
    
    # Zone Annotation
    plt.axvspan(30, 0, color='gray', alpha=0.1, hatch='//')
    plt.text(15, 50, "Collapse Zone", ha='center', va='center', color='#555555')
    
    plt.ylim(-5, 105)
    plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out = os.path.join(os.getcwd(), "Task2_Risk_Curve.png")
    plt.savefig(out, bbox_inches='tight')
    print(f"   [Success] Chart 1 Saved: {out}")

def plot_power_dilution(gammas):
    """ Chart 2: The Grey-Green Curve (Task 2.7) """
    plt.figure(figsize=(9, 5.5), dpi=300)
    
    # Theoretical Model for Bottom-Two Fan Power Retention
    # rho = 1 / (1 + k * gamma)
    k = 0.8
    rhos = 1.0 / (1.0 + k * np.array(gammas))
    
    # Aesthetic Teal Plot
    plt.semilogx(gammas, rhos, marker='o', color=C_B2_MAIN, linewidth=3, 
                 label=r'Fan Power Retention $\rho(\gamma)$', zorder=5)
    
    # Fill with gradient-like shadow
    plt.fill_between(gammas, 0, rhos, color=C_B2_MAIN, alpha=0.15)
    
    # Annotate "Technocrat Zone"
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.6)
    plt.text(1.2, 0.8, "Technocrat Shift\n(Judges Dominate)", fontsize=11, color='#555555')
    
    plt.xlabel(r"Judge Correction Strength ($\gamma$)", fontsize=13, fontweight='bold')
    plt.ylabel(r"Fan Power Retention Ratio", fontsize=13, fontweight='bold')
    plt.title("Bottom-Two Mechanism: Power Dilution Effect", fontsize=15, fontweight='bold', pad=15)
    
    plt.legend(loc='best', frameon=True, fontsize=12)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.ylim(0, 1.05)
    
    out = os.path.join(os.getcwd(), "Task2_Power_Dilution.png")
    plt.savefig(out, bbox_inches='tight')
    print(f"   [Success] Chart 2 Saved: {out}")

# ==========================================
# 5. Main
# ==========================================

def main():
    print("="*60)
    print("Task 2: Generating ALL O-Prize Visuals")
    print("1. Risk S-Curve (Red/Blue)")
    print("2. Power Dilution Curve (Teal/Green)")
    print("="*60)
    
    # 1. Load Data
    loader = DataLoader("Task1_Voting_Inference_Optimized.csv", "2026_MCM_Problem_C_Data.csv")
    try: evidence = loader.load_evidence()
    except Exception as e: print(e); return
    
    # 2. Setup Target
    SEASON = 2
    CONTESTANT = "Jerry Rice"
    
    s2_weeks = [w for (s, w) in evidence.keys() if s == SEASON]
    MAX_DATA_WEEK = max(s2_weeks)
    print(f"\n[System] Target: {CONTESTANT}, Season {SEASON}, Max Week {MAX_DATA_WEEK}")

    # 3. Calc Risk Data
    print(f"\n[Analysis] Simulating Risk Curve...")
    factors = np.linspace(1.2, 0.0, 20) # 20 points
    risk_pct, risk_rank = [], []
    sim = Simulator(evidence)
    
    for f in factors:
        mod_evidence = copy.deepcopy(evidence)
        for k, ev in mod_evidence.items():
            if CONTESTANT in ev.contestant_ids:
                idx = ev.contestant_ids.index(CONTESTANT)
                center = ev.p_ac[idx] * f
                width = (ev.p_max[idx] - ev.p_min[idx]) * 0.5
                ev.p_ac[idx] = center
                ev.p_min[idx] = max(0, center - width)
                ev.p_max[idx] = min(1, center + width)
        
        res_p = sim.run(SEASON, MechanismType.PERCENT, force_elimination=True, n_mc=60)
        res_r = sim.run(SEASON, MechanismType.RANK, force_elimination=True, n_mc=60)
        
        fail_p = np.mean([1 if r.survival_weeks.get(CONTESTANT, 0) < MAX_DATA_WEEK else 0 for r in res_p])
        fail_r = np.mean([1 if r.survival_weeks.get(CONTESTANT, 0) < MAX_DATA_WEEK else 0 for r in res_r])
        
        risk_pct.append(fail_p * 100)
        risk_rank.append(fail_r * 100)
        print(f"   Strength {f*100:3.0f}% -> Pct_Risk: {fail_p*100:3.0f}% | Rank_Risk: {fail_r*100:3.0f}%")

    # 4. Generate Visuals
    print("\n[Plotting] Generating Charts...")
    
    # Chart 1: Risk Curve
    plot_risk_curve(factors, risk_pct, risk_rank, CONTESTANT)
    
    # Chart 2: Power Dilution (The Grey-Green One)
    gammas = np.logspace(-1, 1, 20) # 0.1 to 10
    plot_power_dilution(gammas)
    
    print("\n[Done] All charts generated in current directory.")

if __name__ == "__main__":
    main()