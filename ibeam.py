#!/usr/bin/env python
# coding: utf-8

import os

import matplotlib.pyplot as plt
import numpy as np

# Optuna imports
import optuna
import pandas as pd
import seaborn as sns
import yaml
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.usetex"] = True

# -------------------- Data loading and preprocessing (unchanged) --------------------
df = pd.read_csv("Ratio2.csv")
delta = abs(df['Mn/Mp (FEM)'] - df['Mn/Mp (Formula)'])
threshold = 0.15
df = df[delta <= threshold]
df = df.drop(columns=["Mn/Mp (Formula)"])
df = df.rename(columns={"Mn/Mp (FEM)": "Mn/Mp"})

# df = df[["Lb/ry", "h/tw", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr", "Mn/Mp"]]
df = df[["Lb/ry", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr", "Mn/Mp"]]

tempdf = df.rename(
    columns={
        "Lb/ry": r"L\textsubscript{b}/r\textsubscript{y}",
#        "h/tw": r"h/t\textsubscript{w}",
        "tf/tw": r"t\textsubscript{f}/t\textsubscript{w}",
        "E/Fy": r"E/F\textsubscript{y}",
        "Mp/Mcr": r"M\textsubscript{p}/M\textsubscript{cr}",
        "Mn/Mp": r"M\textsubscript{n}/M\textsubscript{p}",
    },
    errors="raise",
)

# Optional: correlation figures (kept from original script)
plt.figure(figsize=(5, 4))
sns.heatmap(tempdf.corr(numeric_only=True), cmap="YlGnBu", annot=True, fmt='.2f', annot_kws={"fontsize": 8})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/crosscorrelation.png", dpi=640, bbox_inches="tight")
plt.close()

df_stats = df.describe().transpose()
df_stats = df_stats[["mean", "std"]]
df_stats["variance"] = df_stats["std"] ** 2
print(df_stats)

C = tempdf.corr(numeric_only=True)[r"M\textsubscript{n}/M\textsubscript{p}"][:-1]
plt.figure(figsize=(4, 3))
plt.bar(C.index, np.abs(C.values))
plt.ylim([0, 1])
plt.ylabel(r"$|$Correlation with $M\textsubscript{n}/M\textsubscript{p}|$")
plt.savefig("figs/correlation-with-Mn-Mp.png", dpi=640, bbox_inches="tight")
plt.close()

# X_full = df[["Lb/ry", "h/tw", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr"]]
X_full = df[["Lb/ry", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr"]]
y_full = df["Mn/Mp"]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# 5-fold CV
cv5 = KFold(n_splits=5, shuffle=True, random_state=123)


# -------------------- Utility functions (unchanged) --------------------
def evaluate_pipeline(pipeline, threshold=0.2):
    pipeline_train_prd = pipeline.predict(X_train)
    pipeline_train_outliers = np.abs(pipeline_train_prd - y_train) > threshold
    mod_pipeline_train_prd = pipeline_train_prd[~pipeline_train_outliers]
    mod_y_train = y_train[~pipeline_train_outliers]
    pipeline_train_mse = mean_squared_error(mod_pipeline_train_prd, mod_y_train)
    pipeline_train_mae = mean_absolute_error(mod_pipeline_train_prd, mod_y_train)
    pipeline_train_r2_score = r2_score(mod_pipeline_train_prd, mod_y_train)

    pipeline_test_prd = pipeline.predict(X_test)
    pipeline_test_outliers = np.abs(pipeline_test_prd - y_test) > threshold
    mod_pipeline_test_prd = pipeline_test_prd[~pipeline_test_outliers]
    mod_y_test = y_test[~pipeline_test_outliers]
    pipeline_test_mse = mean_squared_error(mod_pipeline_test_prd, mod_y_test)
    pipeline_test_mae = mean_absolute_error(mod_pipeline_test_prd, mod_y_test)
    pipeline_test_r2_score = r2_score(mod_pipeline_test_prd, mod_y_test)

    pipeline_full_prd = pipeline.predict(X_full)
    pipeline_full_outliers = np.abs(pipeline_full_prd - y_full) > threshold
    mod_pipeline_full_prd = pipeline_full_prd[~pipeline_full_outliers]
    mod_y_full = y_full[~pipeline_full_outliers]
    pipeline_full_mse = mean_squared_error(mod_pipeline_full_prd, mod_y_full)
    pipeline_full_mae = mean_absolute_error(mod_pipeline_full_prd, mod_y_full)
    pipeline_full_r2_score = r2_score(mod_pipeline_full_prd, mod_y_full)

    metrics = {
        "y_train": mod_y_train,
        "y_test": mod_y_test,
        "train_mse": pipeline_train_mse,
        "train_mae": pipeline_train_mae,
        "train_r2": pipeline_train_r2_score,
        "test_mse": pipeline_test_mse,
        "test_mae": pipeline_test_mae,
        "test_r2": pipeline_test_r2_score,
        "full_mse": pipeline_full_mse,
        "full_mae": pipeline_full_mae,
        "full_r2": pipeline_full_r2_score,
        "train_predictions": mod_pipeline_train_prd,
        "test_predictions": mod_pipeline_test_prd,
    }
    return metrics


def plot_accuracy(pipeline_metrics, title):
    os.makedirs(title, exist_ok=True)

    print(title)
    print(pipeline_metrics)
    y_train_mod = pipeline_metrics["y_train"]
    y_test_mod = pipeline_metrics["y_test"]

    pipeline_train_prd = pipeline_metrics["train_predictions"]
    pipeline_test_prd = pipeline_metrics["test_predictions"]
    pipeline_train_r2_score = pipeline_metrics["train_r2"]
    pipeline_test_r2_score = pipeline_metrics["test_r2"]
    pipeline_full_r2_score = pipeline_metrics["full_r2"]

    fig_train, axes = plt.subplots(figsize=(2.5, 2.5))
    axes.plot([0, 1], [0, 1])
    axes.text(0.1, 0.7, f"$R^2={pipeline_train_r2_score:.5f}$")
    axes.scatter(
        y_train_mod,
        pipeline_train_prd,
        s=6,
        color="blue",
        alpha=0.5,
        marker='o',
        edgecolors='none',
        label="Train data",
    )
    axes.set_xlabel("Actual Value")
    axes.set_ylabel("Predicted Value")
    axes.legend(loc='lower right')
    fig_train.savefig(f"{title}/{title}-train.png", dpi=640, bbox_inches="tight")
    plt.close(fig_train)

    fig_test, axes = plt.subplots(figsize=(2.5, 2.5))
    axes.plot([0, 1], [0, 1])
    axes.text(0.1, 0.7, f"$R^2={pipeline_test_r2_score:.5f}$")
    axes.scatter(
        y_test_mod,
        pipeline_test_prd,
        s=6,
        color="red",
        alpha=0.5,
        marker='o',
        edgecolors='none',
        label="Test data",
    )
    axes.set_xlabel("Actual Value")
    axes.set_ylabel("Predicted Value")
    axes.legend(loc='lower right')
    fig_test.savefig(f"{title}/{title}-test.png", dpi=640, bbox_inches="tight")
    plt.close(fig_test)

    fig_full, axes = plt.subplots(figsize=(2.5, 2.5))
    axes.plot([0, 1], [0, 1])
    axes.text(0.1, 0.7, f"$R^2={pipeline_full_r2_score:.5f}$")
    axes.scatter(
        y_train_mod,
        pipeline_train_prd,
        s=6,
        color="blue",
        alpha=0.5,
        marker='o',
        edgecolors='none',
        label="Train data",
    )
    axes.scatter(
        y_test_mod,
        pipeline_test_prd,
        s=6,
        color="red",
        alpha=0.5,
        edgecolors='none',
        label="Test data",
    )
    axes.set_xlabel("Actual Value")
    axes.set_ylabel("Predicted Value")
    axes.legend(loc='lower right')
    fig_full.savefig(f"{title}/{title}-full.png", dpi=640, bbox_inches="tight")
    plt.close(fig_full)


# -------------------- Optuna search helpers --------------------
def run_study(
    objective, n_trials=100, study_name="study", direction="minimize", seed=123
):
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction, sampler=sampler, study_name=study_name
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[!] Best value ({direction}): {study.best_value}")
    print("[!] Best params:", study.best_params)
    # Save study to runs folder
    import joblib
    os.makedirs("runs", exist_ok=True)
    joblib.dump(study, f"runs/{study_name}.pkl")
    return study


def neg_mse_cv(model):
    """Returns mean CV negative MSE (to MINIMIZE MSE we will return positive MSE)."""
    scores = cross_val_score(
        model, X_train, y_train, cv=cv5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    # cross_val_score yields NEG-MSE; we want to minimize MSE -> return positive MSE
    return -np.mean(scores)


# -------------------- Random Forest (Optuna + 5-fold CV) --------------------
os.makedirs("random-forest", exist_ok=True)


def rf_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 1000, 5000, step=500)
    criterion = trial.suggest_categorical(
        "criterion", ["squared_error", "friedman_mse", "poisson"]
    )
    max_depth = trial.suggest_int("max_depth", 2, 8)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    model = RandomForestRegressor(
        random_state=42,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )
    return neg_mse_cv(model)


# --- Save best params to YAML ---
besthyp = {}

rf_study = run_study(rf_objective, study_name="rf_opt")
besthyp["RandomForest"] = rf_study.best_params
rf_best = RandomForestRegressor(random_state=42, **rf_study.best_params, n_jobs=-1)
rf_best.fit(X_train, y_train)
rf_metrics = evaluate_pipeline(rf_best, threshold=1e9)
plot_accuracy(rf_metrics, "random-forest")

# -------------------- Gradient Boosting (Optuna + 5-fold CV) --------------------
os.makedirs("gradient-boosting", exist_ok=True)


def gb_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 500, 5000, step=500)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.015, step=0.001)
    max_depth = trial.suggest_int("max_depth", 2, 8)
    model = GradientBoostingRegressor(
        random_state=42,
        subsample=0.7,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
    return neg_mse_cv(model)


gb_study = run_study(gb_objective, study_name="gb_opt")
besthyp["GradientBoosting"] = gb_study.best_params
gb_best = GradientBoostingRegressor(
    random_state=42,
    subsample=0.7,
    **gb_study.best_params
)
gb_best.fit(X_train, y_train)
gb_metrics = evaluate_pipeline(gb_best, threshold=1e9)  # keep original threshold choice
plot_accuracy(gb_metrics, "gradient-boosting")

# feature importance plot
feature_importance = gb_best.feature_importances_
plt.bar(tempdf.columns[:-1], feature_importance)
plt.savefig("gradient-boosting/feature-imp.png", dpi=640, bbox_inches="tight")
plt.close()

# -------------------- SVR (Optuna + Pipeline + 5-fold CV) --------------------
os.makedirs("svr", exist_ok=True)


def make_svr_pipeline(kernel, trial):
    if kernel == "linear":
        C = trial.suggest_float("C", 1e-1, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 0.001, 0.2, log=True)
        svr = SVR(kernel="linear", C=C, epsilon=epsilon)
    elif kernel == "rbf":
        C = trial.suggest_float("C", 1e-1, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 0.001, 0.2, log=True)
        gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
        svr = SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)
    else:  # poly
        C = trial.suggest_float("C", 1e-1, 1e2, log=True)
        epsilon = trial.suggest_float("epsilon", 0.001, 0.2, log=True)
        degree = trial.suggest_int("degree", 2, 5)
        svr = SVR(kernel="poly", C=C, epsilon=epsilon, degree=degree)
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", svr)])
    return pipe


svr_metrics = {}
for kernel in ["linear", "rbf", "poly"]:

    def svr_objective(trial, k=kernel):
        model = make_svr_pipeline(k, trial)
        return neg_mse_cv(model)

    svr_study = run_study(svr_objective, study_name=f"svr_{kernel}_opt")
    if "SVR" not in besthyp:
        besthyp["SVR"] = {}
    besthyp["SVR"][kernel] = svr_study.best_params
    best_pipe = make_svr_pipeline(
        kernel, optuna.trial.FixedTrial(svr_study.best_params)
    )
    best_pipe.fit(X_train, y_train)
    svr_metrics[kernel] = evaluate_pipeline(best_pipe, threshold=1e9)
    plot_accuracy(svr_metrics[kernel], f"SVR-{kernel}")

pd.DataFrame(svr_metrics).filter(regex='_mse', axis=0).transpose().plot.bar()
plt.savefig('svr/svr-kernel-mse.png', dpi=640, bbox_inches='tight')

# -------------------- MLP Regressor (Optuna + 5-fold CV) --------------------
os.makedirs("MLP", exist_ok=True)


def mlp_objective(trial):
    # Allow both "small" and "deeper" networks similar to originals
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden = tuple(
        trial.suggest_int(f"n_units_l{i+1}", 5, 100) for i in range(n_layers)
    )
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
    solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver=solver,
        alpha=alpha,
        random_state=100,
        early_stopping=True,
        max_iter=3000,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    return neg_mse_cv(pipe)


mlp_study = run_study(mlp_objective, study_name="mlp_opt")
besthyp["MLP"] = mlp_study.best_params
# Rebuild the best pipeline from best_params
best_hidden = tuple(
    mlp_study.best_params[f"n_units_l{i+1}"]
    for i in range(mlp_study.best_params["n_layers"])
)
best_mlp = MLPRegressor(
    hidden_layer_sizes=best_hidden,
    activation=mlp_study.best_params["activation"],
    solver=mlp_study.best_params["solver"],
    alpha=mlp_study.best_params["alpha"],
    random_state=100,
    early_stopping=True,
    max_iter=3000,
)
nn_pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", best_mlp)])
nn_pipeline.fit(X_train, y_train)
nn_metrics = evaluate_pipeline(nn_pipeline, threshold=1e9)
plot_accuracy(nn_metrics, "MLP")

# --- Write all best params to YAML file ---
with open("besthyp.yaml", "w") as f:
    yaml.dump(besthyp, f)

# -------------------- Code-vs-Modified comparison section (unchanged) --------------------
# Keep the original vs modified code comparison from your script
try:
    og_data = pd.read_csv("original.csv")
    og_data_ratio = pd.DataFrame()
    og_data_ratio["Lb/ry"] = og_data["Lb"] / og_data["ry"]
    og_data_ratio["h/tw"] = og_data["h0"] / og_data["tw"]
    og_data_ratio["B/2t"] = og_data["b"] / 2 * og_data["tf"]
    og_data_ratio["h/B"] = og_data["h0"] / og_data["b"]
    og_data_ratio["tf/tw"] = og_data["tf"] / og_data["tw"]
    og_data_ratio["E/Fy"] = 29e3 / og_data["Fy (ksi)"]
    og_data_ratio["Mp/Mcr"] = og_data["Mp"] / og_data["Mcr"]
    og_data_ratio["Mn/Mp (AISC)"] = og_data["Mn(AISC code)"] / og_data["Mp"]
    og_data_ratio["Mn/Mp (MOD)"] = og_data["Mn(Modified)"] / og_data["Mp"]

    mse_code = mean_squared_error(
        og_data_ratio["Mn/Mp (AISC)"], og_data_ratio["Mn/Mp (MOD)"]
    )
    r2_code = r2_score(og_data_ratio["Mn/Mp (AISC)"], og_data_ratio["Mn/Mp (MOD)"])
    mae_code = mean_absolute_error(
        og_data_ratio["Mn/Mp (AISC)"], og_data_ratio["Mn/Mp (MOD)"]
    )
    print("Code vs Modified — MSE:", mse_code, "R2:", r2_code, "MAE:", mae_code)
except Exception as e:
    print("Skipping code vs modified comparison due to error:", e)

print("Done. All models tuned with Optuna using 5-fold cross-validation.")
