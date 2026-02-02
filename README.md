# README: The ROBUST-PARETO Mechanism – Implementation for TV Competition Voting System Optimization
## Project Overview
This repository provides the complete implementation of the **ROBUST-PARETO Mechanism**, an innovative analytical framework designed to optimize TV competition voting systems through the integration of **uncertainty quantification** and **feature attribution** techniques. It addresses core challenges in balancing expert evaluations (e.g., judge scores) and audience preferences (e.g., unobserved fan votes) in competitive entertainment contexts, with a focus on robustness, fairness, and audience engagement.

Built on modular, reproducible code, the framework combines convex optimization, Monte Carlo simulation, econometric modeling, and machine learning to achieve four key objectives: (1) infer unobserved audience preferences with quantified uncertainty bounds; (2) compare the performance of existing voting mechanisms; (3) quantify the impact of key factors (e.g., contestant characteristics, partner expertise) on competition outcomes; (4) design a Pareto-optimal voting system that balances conflicting goals (technical merit vs. audience appeal).

All code adheres to high academic standards, with built-in handling of data anomalies, sensitivity analysis, and rigorous validation—meeting the requirements for high-impact empirical and methodological research.  

Notably, `Task1_Voting_Inference_Optimized.csv` is included as a pre-generated attachment. This file contains the output of `task1_code.py`, allowing users to skip the time-consuming execution of Task 1 and directly proceed to subsequent modules.


## File Structure
| Filename                          | Core Functionality                                                                 |
|-----------------------------------|-------------------------------------------------------------------------------------|
| `task1_code.py`                   | Data preprocessing & robust inference of unobserved audience preferences (with uncertainty quantification) |
| `task2_code.py`                   | Counterfactual analysis & performance comparison of two classical voting mechanisms (rank-based vs. percentage-based) |
| `task3_code.py`                   | Feature attribution analysis (quantifies impacts of contestant characteristics, partner effects, and expert scores) |
| `task4_code.py`                   | Design & validation of the ROBUST-PARETO Mechanism (Pareto optimization + robustness auditing) |
| `2026_MCM_Problem_C.csv`          | Input dataset (contestant profiles, expert scores, weekly performance data, and competition outcomes across 34 seasons) |
| `Task1_Voting_Inference_Optimized.csv` | Pre-generated output of Task 1 (inferred audience preferences + uncertainty metrics; avoids long computation time) |


## Dependencies
The framework requires Python 3.8+ with the following libraries (install via `pip install -r requirements.txt`):
```txt
numpy>=1.21.0
pandas>=1.3.5
matplotlib>=3.4.3
seaborn>=0.11.2
scipy>=1.7.3
scikit-learn>=1.0.2
cvxpy>=1.2.2
statsmodels>=0.13.2
xgboost>=1.6.1
shap>=0.41.0
tqdm>=4.64.1
```
- Convex optimization solvers (e.g., `clarabel`, `ecos`, `scs`): Install via `cvxpy.install_solvers()` if missing (required for preference inference in Task 1).  
- Parallel computing support: Enabled by default (uses `cpu_count() - 1` cores), compatible with Windows/macOS/Linux, and accelerates interval estimation and simulation.


## Usage Workflow
Modules are designed to run sequentially, as downstream components (e.g., mechanism design) depend on upstream outputs. Users can leverage the pre-generated `Task1_Voting_Inference_Optimized.csv` to skip Task 1 execution.

### Step 1: Preprocessing & Uncertainty-Aware Preference Inference (`task1_code.py`)
**Purpose**: Clean raw competition data, construct constraints from elimination outcomes, and infer unobserved audience preferences with explicit uncertainty bounds.  

*Note: Skip this step if using the pre-generated `Task1_Voting_Inference_Optimized.csv`.*
```bash
python task1_code.py
```
- **Input**: `2026_MCM_Problem_C.csv` (raw data: contestant characteristics, expert scores, weekly results)  
- **Output**: 
  - `Task1_Voting_Inference_Optimized.csv` (inferred audience preference scores `p_hat` + uncertainty metrics: `p_min`, `p_max`, `uncertainty_width`, `week_uncertainty`)  
  - Diagnostic plots (saved in `Task1_Output_Optimized/`): Distribution of normalized uncertainty, inferred preference share distribution  
- **Key Features**:  
  1. Uncertainty quantification via parallelized linear programming (interval estimation for each preference score)  
  2. Regularized analytic center calculation for robust preference inference within outcome-defined feasible regions  
  3. Expert logic-based handling of structural zeros (e.g., scores for eliminated contestants) and missing data  


### Step 2: Classical Mechanism Comparison (`task2_code.py`)
**Purpose**: Evaluate two classical voting mechanisms (rank-based vs. percentage-based) via counterfactual simulation, generating risk curves and power dilution analyses to identify strengths/weaknesses.
```bash
python task2_code.py
```
- **Input**: 
  1. `Task1_Voting_Inference_Optimized.csv` (pre-generated or from Step 1: inferred preferences + uncertainty)  
  2. `2026_MCM_Problem_C.csv` (raw data for contextual variables)  
- **Output**:  
  1. `Task2_Risk_Curve.png`: S-curve comparing contestant elimination risk across mechanisms (stress-tested under varying audience preference strength)  
  2. `Task2_Power_Dilution.png`: Analysis of audience preference retention under different expert correction strengths  
- **Key Features**:  
  1. Monte Carlo simulation (configurable `n_mc` parameter) to incorporate preference uncertainty into counterfactuals  
  2. Mechanism stress testing (systematic variation of audience preference strength) for robustness assessment  
  3. Adaptive elimination logic (handles weeks with optional/forced eliminations)  


### Step 3: Feature Attribution Analysis (`task3_code.py`)
**Purpose**: Quantify the impact of key factors (contestant age, industry, professional partner expertise, etc.) on competition outcomes using a mixed-methods approach (econometrics + machine learning interpretability).
```bash
python task3_code.py
```
- **Input**: 
  1. `Task1_Voting_Inference_Optimized.csv` (pre-generated or from Step 1: inferred preferences)  
  2. `2026_MCM_Problem_C.csv` (raw data for contestant/partner characteristics)  
- **Output**:  
  1. `Task3_Kingmaker.png`: Partner Effect Analysis (Partner Popularity Contribution Index, PCI) to quantify professional dancers’ impact on audience preferences  
  2. `Task3_SHAP_Summary.png`: Global feature importance (SHAP values) to rank drivers of competition success  
  3. `Task3_Controversy.png`: Case study decomposition (e.g., contestants with high audience support but low expert scores) to identify controversy drivers  
- **Key Features**:  
  1. Econometric elasticity analysis (Mixed Linear Model with OLS fallback) to quantify expert score impact on audience preferences  
  2. Model-agnostic feature attribution (XGBoost + SHAP) for interpretable, robust driver identification  
  3. Integration of partner/contestant characteristics into a unified attribution framework  


### Step 4: ROBUST-PARETO Mechanism Design & Validation (`task4_code.py`)
**Purpose**: Implement and validate the core ROBUST-PARETO Mechanism—an optimized voting system that balances fairness (alignment with expert scores) and attractiveness (audience engagement) via Pareto optimization, with comprehensive robustness auditing.
```bash
python task4_code.py
```
- **Input**: 
  1. `Task1_Voting_Inference_Optimized.csv` (pre-generated or from Step 1: inferred preferences + uncertainty)  
  2. `2026_MCM_Problem_C.csv` (raw data for contextual/characteristic variables)  
- **Output**:  
  1. `Task4_Final_Log.csv`: Simulation results (ROBUST-PARETO Mechanism performance across weeks/seasons)  
  2. `Task4_Validation_ParetoFrontier.png`: Fairness-attractiveness tradeoff plot (validates Pareto optimality vs. classical mechanisms)  
  3. `Task4_Chart1_RobustnessCone_Final.png`: Institutional robustness visualization (shows adaptive response to controversy)  
  4. Sensitivity audit report (printed to console): Assesses performance under parameter uncertainty, vote noise, and system mis-specification  
- **Key Features**:  
  1. Convex feasible sampling to model preference distributions (incorporates uncertainty from Step 1)  
  2. Controversy indexing (integrates judge score variance, expert-audience conflict, and contestant risk factors) with path dependence  
  3. Minimax regret elimination logic to ensure fairness and robustness  
  4. Three-dimensional sensitivity testing (parameter perturbation, vote uncertainty, system error) to validate mechanism stability  


## Methodological Overview
### Core Methodologies Integrating Uncertainty Quantification & Feature Attribution
1. **Robust Convex Inference** (Step 1): Uses linear programming and regularized analytic centers to infer unobserved audience preferences, with uncertainty quantified via interval bounds—ensuring downstream analyses account for estimation error.  
2. **Monte Carlo Counterfactuals** (Step 2): Simulates thousands of competition trajectories using inferred preference distributions (from Step 1) to compare classical mechanisms under uncertainty.  
3. **Mixed-Methods Feature Attribution** (Step 3): Combines econometric elasticity modeling (quantifies linear relationships) and SHAP-based interpretability (captures non-linear/interactive effects) to identify key drivers of competition success—informing the adaptive logic of the ROBUST-PARETO Mechanism.  
4. **Pareto-Optimal Mechanism Design** (Step 4): The ROBUST-PARETO Mechanism integrates:  
   - Uncertainty-aware utility calculation (mean-variance penalty for risk aversion)  
   - Controversy-adaptive weighting (balances expert and audience inputs based on feature attribution insights)  
   - Minimax regret elimination to avoid unfair outcomes (ensures no contestant is eliminated if they could plausibly outperform others, accounting for preference uncertainty)  


### Key Innovations of the ROBUST-PARETO Mechanism
- **Uncertainty Integration**: All components explicitly incorporate preference uncertainty (from Step 1) to avoid overconfident decisions.  
- **Feature-Driven Adaptivity**: Controversy indexing and utility weighting are informed by feature attribution results (from Step 3), ensuring the mechanism responds to context-specific factors (e.g., high-risk contestants, expert-audience conflicts).  
- **Pareto Optimality**: Balances fairness (alignment with expert technical scores) and attractiveness (retention of audience-favorite contestants) without compromising either objective.  
- **Robustness by Design**: Built-in handling of numerical instability (auto-fallback methods) and systematic sensitivity testing to ensure performance across diverse competition scenarios.  


## Notes for Academic Use
1. **Data Path Configuration**: Modify the `_find_file()` method in `DataLoader` (Step 1), `DataIntegrator` (Step 3), and `DataEngine` (Step 4) to match your local file structure.  
2. **Parameter Tuning**: Key hyperparameters are configurable to adapt to different competition contexts:  
   - Uncertainty quantification: `lambda_reg` (regularization strength in Step 1)  
   - Simulation rigor: `n_mc` (Monte Carlo samples in Step 2) and `MC_SAMPLES` (Step 4)  
   - Mechanism behavior: `RHO` (risk aversion coefficient) and `BETA` (controversy sensitivity) in Step 4  
3. **Computational Efficiency**: Parallel processing is enabled by default (uses `cpu_count() - 1` cores). Adjust `n_procs` in Step 1 for resource-constrained environments.  
4. **Visualization**: All plots are optimized for academic publication (serif fonts, high DPI = 300, consistent color palettes) and can be modified via the `set_style()` functions in each script.  


---
*Last Updated: 2026/2/2*  
