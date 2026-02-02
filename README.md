# Group2612508-Code-Appendix
# README: The ROBUST-PARETO Mechanism – Implementation for TV Competition Voting System Optimization
## Project Overview
This repository presents the full implementation of the **ROBUST-PARETO Mechanism**, a novel analytical framework designed for optimizing TV competition voting systems through the integration of **uncertainty quantification** and **feature attribution**. The framework addresses critical challenges in balancing expert evaluations (e.g., judge scores) and audience preferences (e.g., unobserved fan votes) in competitive entertainment contexts, with a focus on robustness, fairness, and stakeholder engagement.

Built on modular, reproducible code, the framework combines convex optimization, Monte Carlo simulation, econometric modeling, and machine learning to: (1) infer unobserved audience preferences with uncertainty bounds; (2) compare existing voting mechanisms; (3) quantify the impact of key factors (e.g., participant characteristics, partner expertise) on competition outcomes; and (4) design a Pareto-optimal voting system that balances conflicting objectives (technical merit vs. audience appeal).

All code adheres to high academic standards, with built-in handling of data irregularities, sensitivity analysis, and rigorous validation—aligning with the requirements for impactful empirical and methodological research.

## File Structure
| Filename               | Core Functionality                                                                 |
|------------------------|-------------------------------------------------------------------------------------|
| `task1_code.py`        | Data preprocessing & robust inference of unobserved audience preferences (with uncertainty quantification) |
| `task2_code.py`        | Counterfactual analysis & comparison of two classical voting mechanisms (rank-based vs. percentage-based) |
| `task3_code.py`        | Feature attribution analysis (participant characteristics, partner effects, expert score impact) |
| `task4_code.py`        | Design & validation of the ROBUST-PARETO Mechanism (Pareto optimization + robustness auditing) |
| `2026_MCM_Problem_C.csv` | Input dataset (participant profiles, expert scores, weekly performance data, and competition outcomes across 34 seasons) |

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
- Convex optimization solvers (e.g., `clarabel`, `ecos`, `scs`): Install via `cvxpy.install_solvers()` if missing (required for preference inference).
- Parallel computing support: Utilizes the `multiprocessing` library (compatible with Windows/macOS/Linux) to accelerate interval estimation and simulation.

## Usage Workflow
The modules are designed to be executed sequentially, as downstream components (e.g., mechanism design) depend on upstream outputs (e.g., inferred preferences).

### Step 1: Preprocessing & Uncertainty-Aware Preference Inference (`task1_code.py`)
**Purpose**: Clean raw competition data, construct constraints from elimination outcomes, and infer unobserved audience preferences using robust convex feasible region analysis—with explicit quantification of uncertainty (interval bounds for each estimate).

```bash
python task1_code.py
```
- **Input**: `2026_MCM_Problem_C.csv` (raw data: participant characteristics, expert scores, weekly results)
- **Output**: 
  - `Task1_Voting_Inference_Optimized.csv`: Inferred audience preference scores (`p_hat`) + uncertainty metrics (`p_min`, `p_max`, `uncertainty_width`, `week_uncertainty`).
  - Diagnostic plots (saved in `Task1_Output_Optimized/`): Distribution of normalized uncertainty, inferred preference share distribution.
- **Key Features Aligned with Framework Goals**:
  - Uncertainty quantification via parallelized linear programming (interval estimation for each preference score).
  - Regularized analytic center calculation to robustly infer preferences within feasible regions defined by competition outcomes.
  - Expert logic-based handling of structural zeros (e.g., scores for eliminated participants) and missing data.

### Step 2: Classical Mechanism Comparison (`task2_code.py`)
**Purpose**: Evaluate the performance of two classical voting mechanisms (rank-based vs. percentage-based) via counterfactual simulation, generating risk curves and power dilution analyses to identify strengths/weaknesses.

```bash
python task2_code.py
```
- **Input**: 
  - `Task1_Voting_Inference_Optimized.csv` (inferred preferences + uncertainty from Step 1)
  - `2026_MCM_Problem_C.csv` (raw data for contextual variables)
- **Output**:
  - `Task2_Risk_Curve.png`: S-curve comparing elimination risk for contestants across mechanism types (stress-tested under varying audience preference strength).
  - `Task2_Power_Dilution.png`: Analysis of audience preference retention under different expert correction strengths.
- **Key Features Aligned with Framework Goals**:
  - Monte Carlo simulation (configurable `n_mc` parameter) to account for preference uncertainty in counterfactuals.
  - Mechanism stress testing (systematic variation of audience preference strength) to assess robustness.
  - Adaptive elimination logic (handles weeks with optional/forced eliminations).

### Step 3: Feature Attribution Analysis (`task3_code.py`)
**Purpose**: Quantify the impact of key factors (participant age, industry, professional partner expertise, etc.) on competition outcomes using a mixed-methods approach (econometrics + machine learning interpretability).

```bash
python task3_code.py
```
- **Input**: 
  - `Task1_Voting_Inference_Optimized.csv` (inferred preferences from Step 1)
  - `2026_MCM_Problem_C.csv` (raw participant/partner characteristic data)
- **Output**:
  - `Task3_Kingmaker.png`: Partner Effect Analysis (Partner Popularity Contribution Index, PCI) to quantify professional dancers’ impact on audience preferences.
  - `Task3_SHAP_Summary.png`: Global feature importance (SHAP values) to rank drivers of competition success.
  - `Task3_Controversy.png`: Case study decomposition (e.g., high-audience/low-expert-score contestants) to identify controversy drivers.
- **Key Features Aligned with Framework Goals**:
  - Econometric elasticity analysis (Mixed Linear Model with OLS fallback) to quantify expert score impact on audience preferences.
  - Model-agnostic feature attribution (XGBoost + SHAP) for interpretable, robust driver identification.
  - Integration of partner/participant characteristics into a unified attribution framework.

### Step 4: ROBUST-PARETO Mechanism Design & Validation (`task4_code.py`)
**Purpose**: Implement and validate the core ROBUST-PARETO Mechanism—an optimized voting system that balances fairness (alignment with expert scores) and attractiveness (audience engagement) via Pareto optimization, with comprehensive robustness auditing.

```bash
python task4_code.py
```
- **Input**: 
  - `Task1_Voting_Inference_Optimized.csv` (inferred preferences + uncertainty from Step 1)
  - `2026_MCM_Problem_C.csv` (raw data for contextual/characteristic variables)
- **Output**:
  - `Task4_Final_Log.csv`: Simulation results (ROBUST-PARETO Mechanism performance across weeks/seasons).
  - `Task4_Validation_ParetoFrontier.png`: Fairness-attractiveness tradeoff plot (validates Pareto optimality vs. classical mechanisms).
  - `Task4_Chart1_RobustnessCone_Final.png`: Institutional robustness visualization (shows adaptive response to controversy).
  - Sensitivity audit report (printed to console): Assesses performance under parameter uncertainty, vote noise, and system mis-specification.
- **Key Features Aligned with Framework Goals**:
  - Convex feasible sampling to model preference distributions (accounting for uncertainty from Step 1).
  - Controversy indexing (integrates judge variance, expert-audience conflict, and participant risk factors) with path dependence.
  - Minimax regret elimination logic to ensure fairness and robustness.
  - Three-dimensional sensitivity testing (parameter perturbation, vote uncertainty, system error) to validate mechanism stability.

## Methodological Overview
### Core Methodologies Integrating Uncertainty Quantification & Feature Attribution
1. **Robust Convex Inference** (Task 1): Uses linear programming and regularized analytic centers to infer unobserved audience preferences while quantifying uncertainty via interval bounds—ensuring downstream analyses account for estimation error.
2. **Monte Carlo Counterfactuals** (Task 2): Simulates thousands of competition trajectories using inferred preference distributions (from Step 1) to compare classical mechanisms under uncertainty.
3. **Mixed-Methods Feature Attribution** (Task 3): Combines econometric elasticity modeling (quantifies linear relationships) and SHAP-based interpretability (captures non-linear/交互 effects) to identify key drivers of competition success—feeding into the ROBUST-PARETO Mechanism’s adaptive logic.
4. **Pareto-Optimal Mechanism Design** (Task 4): The ROBUST-PARETO Mechanism integrates:
   - Uncertainty-aware utility calculation (mean-variance penalty for risk aversion).
   - Controversy-adaptive weighting (balances expert and audience inputs based on feature attribution insights).
   - Minimax regret elimination to avoid unfair outcomes—ensuring no contestant is eliminated if they could plausibly outperform others (accounting for preference uncertainty).

### Key Innovations of the ROBUST-PARETO Mechanism
- **Uncertainty Integration**: All components explicitly incorporate preference uncertainty (from Step 1) to avoid overconfident decisions.
- **Feature-Driven Adaptivity**: Controversy indexing and utility weighting are informed by feature attribution results (from Step 3), ensuring the mechanism responds to context-specific factors (e.g., high-risk participants, expert-audience conflicts).
- **Pareto Optimality**: Balances fairness (alignment with expert technical scores) and attractiveness (retention of audience-favorite contestants) without compromising either objective.
- **Robustness by Design**: Built-in handling of numerical instability (auto-fallback methods) and systematic sensitivity testing to ensure performance across diverse competition scenarios.

*Last Updated: 2026/2/2*
