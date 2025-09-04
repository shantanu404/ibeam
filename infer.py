import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

# Load best hyperparameters from YAML
with open('besthyp.yaml', 'r') as f:
	besthyp = yaml.safe_load(f)

# Load training data (same columns as in ibeam.py)
df = pd.read_csv('Ratio2.csv')
delta = abs(df['Mn/Mp (FEM)'] - df['Mn/Mp (Formula)'])
threshold = 0.15
df = df[delta <= threshold]
df = df.drop(columns=["Mn/Mp (Formula)"])
df = df.rename(columns={"Mn/Mp (FEM)": "Mn/Mp"})

X_full = df[["Lb/ry", "h/tw", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr"]]
y_full = df["Mn/Mp"]

# Load inference data
infer_df = pd.read_csv('inference.csv')
print(infer_df.head())
# Ensure columns match training features
X_infer = infer_df[["Lb/ry", "h/tw", "B/2t", "h/B", "tf/tw", "E/Fy", "Mp/Mcr"]]

# Train GradientBoostingRegressor on full data (no scaling)
gb_params = besthyp["GradientBoosting"]
gb_model = GradientBoostingRegressor(
	random_state=42,
	subsample=0.7,
	**gb_params
)
gb_model.fit(X_full, y_full)

# Infer values for inference.csv
inferred = gb_model.predict(X_infer)
infer_df["Mn/Mp_pred"] = inferred
infer_df["Mn_pred"] = inferred * infer_df["Mp"]

# Save results
infer_df.to_csv("inference_with_pred.csv", index=False)
print("Inference complete. Results saved to inference_with_pred.csv.")

# --- TSNE Visualization ---
all_X = pd.concat([X_full, X_infer], ignore_index=True)
all_y = np.concatenate([y_full, inferred])
is_train = np.array([True]*len(X_full) + [False]*len(X_infer))


tsne = TSNE(n_components=2, random_state=42)
all_X_embedded = tsne.fit_transform(all_X)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Train only
sc1 = axes[0].scatter(
	all_X_embedded[is_train,0],
	all_X_embedded[is_train,1],
	c=all_y[is_train],
	cmap='viridis',
	marker='D',
	edgecolor='none',
	alpha=0.7
)
cb1 = fig.colorbar(sc1, ax=axes[0])
cb1.set_label('Train Output (Mn/Mp)')
axes[0].set_title('TSNE: Train Data Only')

# Inference only
sc2 = axes[1].scatter(
	all_X_embedded[~is_train,0],
	all_X_embedded[~is_train,1],
	c=all_y[~is_train],
	cmap='viridis',
	marker='o',
	edgecolor='none',
	alpha=0.7
)
cb2 = fig.colorbar(sc2, ax=axes[1])
cb2.set_label('Inference Output (Mn/Mp_pred)')
axes[1].set_title('TSNE: Inference Data Only')

for ax in axes:
	ax.set_xticks([])
	ax.set_yticks([])

plt.tight_layout()
plt.savefig('tsne_train_infer_subplot.png', dpi=300)
# plt.show()
