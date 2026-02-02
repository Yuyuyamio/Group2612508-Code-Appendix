import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import re
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from multiprocessing import Pool, cpu_count
import functools

# ==========================================
# 0. Global Settings and Helper Functions
# ==========================================
# Set plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
warnings.filterwarnings('ignore')

def _worker_solve_lp(args):
    """
    Subtask function for parallel solving of Linear Programming (LP) problems.
    Defined at the global level to support multiprocessing pickle on Windows.
    """
    idx, sense, c_val, A_ub, b_ub, A_eq, b_eq, bounds = args
    
    # Construct objective vector
    n = A_ub.shape[1]
    c = np.zeros(n)
    c[idx] = c_val # minimize c^T x
    
    try:
        # Use the 'highs' method, which is the fastest LP solver in scipy currently
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                      bounds=bounds, method='highs')
        
        # If sense is -1 (maximization), invert the result
        val = -res.fun if sense == -1 else res.fun
        return (idx, sense, val if res.success else None)
    except Exception:
        return (idx, sense, None)

# ==========================================
# Module 1: Data Preprocessing
# ==========================================
class DancingDataPreprocessor:
    """
    Responsible for data cleaning, restructuring, and feature engineering.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.long_data = None
        self.processed_data = None
        self.elim_week_dict = {} 
        self.season_max_weeks = {}
        
    def load_and_process(self):
        print("[Preprocessing] Loading data...")
        self.raw_data = pd.read_csv(self.file_path)
        
        # 1. Precompute the maximum number of weeks for each season
        self.season_max_weeks = self._calculate_season_max_weeks()
        
        # 2. Parse elimination weeks
        print("[Preprocessing] Parsing elimination logic...")
        self._build_elimination_map()
        
        # 3. Convert to long format data
        print("[Preprocessing] Melting data structures...")
        self._melt_data()
        
        # 4. Handle zero values
        print("[Preprocessing] Handling structural zeros (Expert Logic)...")
        self._handle_zeros()
        
        # 5. [Key Fix] Construct final features (J_share, is_eliminated)
        final_df = self.get_weekly_contestants()
        
        return final_df

    def _calculate_season_max_weeks(self):
        max_weeks = {}
        if 'season' not in self.raw_data.columns: return {}
        seasons = self.raw_data['season'].unique()
        for s in seasons:
            season_data = self.raw_data[self.raw_data['season'] == s]
            max_w = 0
            for col in season_data.columns:
                match = re.search(r'week(\d+)_judge', col)
                if match:
                    week_num = int(match.group(1))
                    if season_data[col].replace(0, np.nan).notna().any():
                        max_w = max(max_w, week_num)
            max_weeks[s] = max_w
        return max_weeks

    def _build_elimination_map(self):
        for idx, row in self.raw_data.iterrows():
            key = (row['season'], row['celebrity_name'])
            res = row['results']
            
            if not isinstance(res, str):
                self.elim_week_dict[key] = np.inf
                continue
                
            if 'Eliminated Week' in res:
                try:
                    week = int(re.search(r'Week (\d+)', res).group(1))
                    self.elim_week_dict[key] = week
                except:
                    self.elim_week_dict[key] = np.inf
            elif re.search(r'(Winner|Runner-up|Place|Finalist)', res, re.IGNORECASE):
                self.elim_week_dict[key] = self.season_max_weeks.get(row['season'], 10)
            elif 'Withdrew' in res:
                match = re.search(r'Week (\d+)', res)
                self.elim_week_dict[key] = int(match.group(1)) if match else -1
            else:
                self.elim_week_dict[key] = np.inf

    def _melt_data(self):
        id_vars = ['celebrity_name', 'season']
        score_cols = [c for c in self.raw_data.columns if 'judge' in c]
        
        self.long_data = self.raw_data.melt(
            id_vars=id_vars, 
            value_vars=score_cols, 
            var_name='week_judge', 
            value_name='score'
        )
        
        pattern = r'week(\d+)_judge(\d+)_score'
        extracted = self.long_data['week_judge'].str.extract(pattern)
        self.long_data['week'] = extracted[0].astype(int)
        self.long_data['judge'] = extracted[1].astype(int)
        self.long_data['score'] = pd.to_numeric(self.long_data['score'], errors='coerce')

    def _handle_zeros(self):
        df = self.long_data.copy()
        # Vectorized calculation of elim_week
        df['elim_week'] = df.apply(lambda x: self.elim_week_dict.get((x['season'], x['celebrity_name']), np.inf), axis=1)
        
        # Structural zero logic: week > elimination week => NaN
        condition_structural = (df['elim_week'] != -1) & (df['week'] > df['elim_week'])
        condition_withdrew = (df['elim_week'] == -1)
        mask_structural = condition_structural | condition_withdrew
        
        df.loc[mask_structural, 'score'] = np.nan
        
        # Calculate mean values
        stats = df.groupby(['season', 'week', 'celebrity_name'])['score'].agg(['mean', 'count']).reset_index()
        stats.rename(columns={'mean': 'avg_score', 'count': 'valid_judges'}, inplace=True)
        
        self.processed_data = stats
        
    def get_weekly_contestants(self):
        print("[Preprocessing] Constructing final constraint datasets...")
        if self.processed_data is None:
            raise ValueError("Processed data is empty. Run _handle_zeros first.")
            
        df = self.processed_data.copy()
        
        df['elim_week'] = df.apply(lambda x: self.elim_week_dict.get((x['season'], x['celebrity_name']), np.inf), axis=1)
        df = df.dropna(subset=['avg_score'])
        
        # Calculate J_share (key step)
        sums = df.groupby(['season', 'week'])['avg_score'].transform('sum')
        df['J_share'] = df['avg_score'] / sums
        
        # Mark if eliminated in the current week
        df['is_eliminated'] = (df['week'] == df['elim_week'])
        
        return df

# ==========================================
# Module 2: Enhanced RCVP Model (Robust Inference)
# ==========================================
class RCVPInferenceModel:
    """
    Task 1 Model: Robust Convex Feasible Region Inference.
    Optimized with Regularized Analytic Center and Parallel Interval Estimation.
    """
    def __init__(self, processed_data):
        self.data = processed_data
        self.results = []

    def build_sparse_constraints(self, n, elim_indices, safe_indices, J):
        """Build sparse constraint matrices A and b"""
        rows, cols, data = [], [], []
        b_vals = []
        
        row_idx = 0
        for e_idx in elim_indices:
            for s_idx in safe_indices:
                # Constraint: p_e - p_s <= J_s - J_e
                rows.extend([row_idx, row_idx])
                cols.extend([e_idx, s_idx])
                data.extend([1.0, -1.0])
                
                bound = J[s_idx] - J[e_idx]
                b_vals.append(bound)
                row_idx += 1
        
        if row_idx == 0:
            return None, None
            
        A_sparse = csr_matrix((data, (rows, cols)), shape=(row_idx, n))
        b_arr = np.array(b_vals)
        return A_sparse, b_arr

    def solve_season_week(self, season, week, group_df):
        dancers = group_df['celebrity_name'].tolist()
        n = len(dancers)
        # J_share has been correctly generated by Preprocessor
        J = group_df['J_share'].values 
        
        eliminated_mask = group_df['is_eliminated'].values
        
        # Case 1: No Elimination
        if not np.any(eliminated_mask):
            self._record_result(season, week, dancers, np.ones(n)/n, np.zeros(n), np.ones(n), "No Elimination")
            return

        # Case 2: Elimination Constraints
        elim_indices = np.where(eliminated_mask)[0]
        safe_indices = np.where(~eliminated_mask)[0]
        
        if len(safe_indices) == 0: return 
        
        A, b = self.build_sparse_constraints(n, elim_indices, safe_indices, J)
        if A is None: return

        # Solve regularized analytic center
        p_ac, status = self._solve_regularized_center(n, A, b)
        
        if status == "Failed":
             self._record_result(season, week, dancers, np.ones(n)/n, np.zeros(n), np.ones(n), "Infeasible")
             return

        # Parallel interval estimation
        p_min, p_max = self._solve_interval_parallel(n, A, b, p_ac, status)
        
        self._record_result(season, week, dancers, p_ac, p_min, p_max, status)

    def _solve_regularized_center(self, n, A, b):
        p = cp.Variable(n)
        m = A.shape[0]
        slack = cp.Variable(m, nonneg=True)
        
        constraints = [cp.sum(p) == 1, p >= 1e-6]
        constraints += [A @ p <= b + slack]
        
        lambda_reg = 0.5
        barrier = cp.sum(cp.log(b + slack - A @ p))
        regularization = lambda_reg * cp.sum_squares(slack)
        
        objective = cp.Maximize(barrier - regularization)
        prob = cp.Problem(objective, constraints)
        
        solvers_to_try = [
            (cp.CLARABEL, {}), 
            (cp.ECOS, {}),     
            (cp.SCS, {'max_iters': 10000}) 
        ]
        
        if 'MOSEK' in cp.installed_solvers():
            solvers_to_try.insert(0, (cp.MOSEK, {}))
            
        final_status = "Failed"
        p_val = None
        
        for solver, kwargs in solvers_to_try:
            try:
                prob.solve(solver=solver, verbose=False, **kwargs)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    p_val = p.value
                    if np.sum(slack.value) > 1e-3:
                        final_status = "Relaxed"
                    else:
                        final_status = "Feasible"
                    break
            except Exception:
                continue
                
        if p_val is None:
            return np.ones(n)/n, "Failed"
            
        return p_val, final_status

    def _solve_interval_parallel(self, n, A, b, p_ac, status):
        current_b = b
        if status == "Relaxed":
            residuals = A @ p_ac - b
            max_viol = np.max(np.maximum(0, residuals))
            current_b = b + max_viol + 1e-5
            
        if hasattr(A, 'toarray'):
            A_dense = A.toarray()
        else:
            A_dense = A
            
        A_ub = np.vstack([A_dense, -np.eye(n)])
        b_ub = np.concatenate([current_b, np.zeros(n)])
        A_eq = np.ones((1, n))
        b_eq = np.array([1])
        bounds = (0, 1)
        
        tasks = []
        for i in range(n):
            tasks.append((i, -1, -1.0, A_ub, b_ub, A_eq, b_eq, bounds)) # Max
            tasks.append((i, 1, 1.0, A_ub, b_ub, A_eq, b_eq, bounds))   # Min
            
        n_procs = max(1, cpu_count() - 1)
        
        try:
            with Pool(processes=n_procs) as pool:
                results = pool.map(_worker_solve_lp, tasks)
        except Exception as e:
            print(f"Parallel fallback: {e}")
            results = map(_worker_solve_lp, tasks)
            
        p_min = np.zeros(n)
        p_max = np.zeros(n)
        
        for idx, sense, val in results:
            if val is None: val = p_ac[idx]
            if sense == -1:
                p_max[idx] = val
            else:
                p_min[idx] = val
                
        return p_min, p_max

    def _record_result(self, season, week, dancers, p_est, p_min, p_max, status):
        n = len(dancers)
        if p_est is None: p_est = np.full(n, 1.0/n)
        if p_min is None: p_min = np.zeros(n)
        if p_max is None: p_max = np.ones(n)

        width = p_max - p_min
        # Simple average uncertainty, can be replaced with entropy metric if needed
        normalized_uncertainty = np.mean(width) 
        
        for i in range(n):
            self.results.append({
                'season': season,
                'week': week,
                'dancer': dancers[i],
                'p_hat': p_est[i],
                'p_min': p_min[i],
                'p_max': p_max[i],
                'uncertainty_width': width[i],
                'week_uncertainty': normalized_uncertainty,
                'status': status
            })

    def run_inference(self):
        print("[RCVP] Running Robust Convex Feasible Region Inference (Parallelized)...")
        grouped = self.data.groupby(['season', 'week'])
        groups = list(grouped)
        
        for (season, week), group in tqdm(groups):
            if len(group) < 2: continue
            self.solve_season_week(season, week, group)
            
        return pd.DataFrame(self.results)

# ==========================================
# Module 3: Result Analyzer
# ==========================================
class RCVPResultAnalyzer:
    def __init__(self, results_df):
        self.df = results_df
        if not os.path.exists('Task1_Output_Optimized'):
            os.makedirs('Task1_Output_Optimized')

    def analyze_consistency(self):
        print("\n[Analysis] Calculating Optimized Consistency Metrics...")
        week_stats = self.df.groupby(['season', 'week'])['status'].first()
        counts = week_stats.value_counts()
        total = len(week_stats)
        
        print("Consistency Report:")
        for status, count in counts.items():
            print(f"  {status}: {count} ({count/total:.2%})")

    def plot_charts(self):
        print("[Analysis] Generating Optimization Charts...")
        # 1. Uncertainty Boxplot
        plt.figure(figsize=(14, 6))
        plot_data = self.df[self.df['status'].str.contains('Feasible|Relaxed', na=False)]
        if not plot_data.empty:
            sns.boxplot(data=plot_data, x='season', y='week_uncertainty', color='skyblue', fliersize=1)
            plt.title('Distribution of Normalized Uncertainty Across Seasons')
            plt.savefig('Task1_Output_Optimized/Fig_Uncertainty.png')
        
        # 2. Vote Share Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['p_hat'], bins=50, kde=True)
        plt.title('Distribution of Inferred Vote Shares ($\hat{p}$)')
        plt.xlabel('Vote Share')
        plt.savefig('Task1_Output_Optimized/Fig_VoteShare_Dist.png')

# ==========================================
# Main Program Entry
# ==========================================
def main():
    file_path = '2026_MCM_Problem_C_Data.csv' 
    
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    # 1. Data Preprocessing
    preprocessor = DancingDataPreprocessor(file_path)
    input_data = preprocessor.load_and_process()
    
    # Print first few rows to check if J_share exists
    print("\n[Preview] Data ready for inference:")
    print(input_data[['season', 'week', 'celebrity_name', 'J_share']].head())
    
    # 2. Model Inference
    model = RCVPInferenceModel(input_data)
    results_df = model.run_inference()
    
    # 3. Save and Analyze
    results_df.to_csv("Task1_Voting_Inference_Optimized.csv", index=False)
    
    analyzer = RCVPResultAnalyzer(results_df)
    analyzer.analyze_consistency()
    analyzer.plot_charts()
    
    print("\n[Success] Optimized Inference Complete. Results saved.")

if __name__ == "__main__":
    main()