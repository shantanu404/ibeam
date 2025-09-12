import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna

# Set matplotlib rcParams as in ibeam.py
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True

def plot_best_performance_across_hyperparams(cv_result, titlename):
    """
    Plots the best (minimum) test score for each value of every hyperparameter in cv_result['params'].
    Saves a polar plot for each hyperparameter in the titlename folder.
    """
    params_df = pd.DataFrame(cv_result['params'])
    mean_mse = np.array(cv_result['mean_test_score'])
    heatmap_data = params_df.copy()
    heatmap_data['mean_r2'] = mean_mse
    if all(mean_mse < 0):
        heatmap_data['mean_r2'] = -mean_mse

    os.makedirs(titlename, exist_ok=True)
    for col in params_df.columns:
        df = heatmap_data.groupby(col)['mean_r2'].min()
        metrics = df.index.tolist()
        values = df.values.tolist()
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))  # Close the loop
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.margins(0.1)
        ax.scatter(angles, values)
        ax.plot(angles, values, '--')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        # Check if labels are floats and format them
        try:
            float_labels = [float(label) for label in metrics]
            ax.set_xticklabels([f"{label:.2f}" for label in float_labels])
        except ValueError:
            pass  # Labels are not all floats, keep original

        plt.title(f"Best performance across $\\texttt{{{col}}}$")
        plt.savefig(f"{titlename}/{titlename}-hyperparam-{col}.png", dpi=640, bbox_inches="tight")
        plt.close(fig)

# Example usage:
# cv_result = pd.read_pickle('runs/rf_opt_cv_result.pkl')
# plot_best_performance_across_hyperparams(cv_result, 'random-forest')

# Load Optuna studies and plot best performance across hyperparameters
def optuna_study_to_cv_result(study):
    """
    Convert Optuna study trials to a cv_result-like dict for plotting.
    Returns a dict with 'params' and 'mean_test_score'.
    """
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    params = [t.params for t in trials]
    scores = [t.value for t in trials]
    return {'params': params, 'mean_test_score': scores}

def plot_all_optuna_runs():
    runs_dir = 'runs'
    for fname in os.listdir(runs_dir):
        if fname.endswith('.pkl'):
            study_name = fname.replace('.pkl', '')
            study = joblib.load(os.path.join(runs_dir, fname))
            cv_result = optuna_study_to_cv_result(study)
            plot_best_performance_across_hyperparams(cv_result, study_name)

plot_all_optuna_runs()