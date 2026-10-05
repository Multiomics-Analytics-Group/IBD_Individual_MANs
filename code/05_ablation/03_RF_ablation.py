import pandas as pd
import numpy as np
import copy
import multiprocessing
import pickle
from pathlib import Path
from scipy.stats import gmean
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path.cwd()
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/IBD_CuratedMetagenomicData"
RESULTS_PATH = PROJECT_ROOT / "results/machine_learning/gnn_top20_graphnodes"
TOP_NODES_PATH = PROJECT_ROOT / "results/tables/analyses/attributions/top_20_graphnodes.csv"
METRICS_PATH = PROJECT_ROOT / "results/tables/analyses/attributions"

FEATURE_CONFIG = {
    "include_abundance": True,           # Master toggle for species abundance features
    "use_top20_gnn": True,              # Filter taxa to Top 20 GNN nodes (if include_abundance=True)
    "include_density": False,             # Include network density metric
    "include_pos_edge_prop": False,       # Include proportion of positive edges metric
    "include_mean_strength": False,        # Include mean edge weight metric
    "include_transitivity": False,        # Include transitivity metric
    "include_degree_norm": False,         # Include normalized degree metric
    "include_betweenness_norm": False,   # Include normalized betweenness metric
    "include_ecoli_kpneumo_w": False      # Include absolute E. coli - K. pneumoniae edge weight
}

# Load abundance data
composite_all = pd.read_csv(PROCESSED_DATA_PATH / "HMP_counts_filtered_composite.csv", index_col="id")
counts_species = composite_all[[col for col in composite_all.columns if col.startswith("k__")]]
subject_ids = composite_all["subject_id"]

composite_all_Hall = pd.read_csv(PROCESSED_DATA_PATH / "Hall_counts_filtered_composite.csv", index_col="id")
counts_species_Hall = composite_all_Hall[[col for col in composite_all_Hall.columns if col.startswith("k__")]]
subject_ids_Hall = composite_all_Hall["subject_id"]

# Match taxa columns between training and external datasets
common_taxa = counts_species.columns.intersection(counts_species_Hall.columns)
counts_species_Hall = counts_species_Hall[common_taxa]

missing_taxa = counts_species.columns.difference(counts_species_Hall.columns)
for taxon in missing_taxa:
    counts_species_Hall[taxon] = 0

counts_species_Hall = counts_species_Hall[counts_species.columns]


# Preprocessing functions
def clr(df, pseudocount=1):
    df = df + pseudocount
    gm = gmean(df, axis=1)
    return np.log(df.div(gm, axis=0))

def drop_unclassified(df, rank_prefix='s__'):
    cols_to_keep = []
    for col in df.columns:
        segments = col.split('|')
        keep = True
        for seg in segments:
            if seg.startswith(rank_prefix) and 'unclassified' in seg[len(rank_prefix):].lower():
                keep = False
                break
        if keep:
            cols_to_keep.append(col)
    return df[cols_to_keep]


#Process datasets and filter timepoints
clr_counts_species = clr(counts_species)
summary_counts_species = clr_counts_species.groupby(subject_ids).mean()

clr_counts_species_Hall = clr(counts_species_Hall)
summary_counts_species_Hall = clr_counts_species_Hall.groupby(subject_ids_Hall).mean()

# Keep subjects with at least 7 timepoints
valid_subjects = subject_ids.value_counts()[lambda x: x >= 7].index
summary_counts_species = summary_counts_species.loc[valid_subjects]

valid_subjects_Hall = subject_ids_Hall.value_counts()[lambda x: x >= 7].index
summary_counts_species_Hall = summary_counts_species_Hall.loc[valid_subjects_Hall]

# Remove unclassified taxa
summary_counts_species = drop_unclassified(summary_counts_species, 's__')
summary_counts_species_Hall = drop_unclassified(summary_counts_species_Hall, 's__')


# Modular function to assemble feature matrix based on configuration toggles
def assemble_feature_matrix(df_hmp, df_hall, config):

    # --- Abundance Feature Toggle ---
    if config.get("include_abundance", True):
        # Handle subsetting (Top 20 vs All) if abundance is enabled
        if config.get("use_top20_gnn", False):
            top_20_nodes = pd.read_csv(TOP_NODES_PATH)["x"].tolist()
            
            def get_clean_species_name(col_name):
                clean = col_name.split("s__")[-1] if "s__" in col_name else col_name
                return clean.replace("_", " ")

            column_mapping = {get_clean_species_name(col): col for col in df_hmp.columns}
            selected_columns = [column_mapping[node] for node in top_20_nodes if node in column_mapping]
            
            print(f"[CONFIG] Including Top 20 GNN Abundance Taxa ({len(selected_columns)} matched).")
            hmp_out = df_hmp[selected_columns].copy()
            hall_out = df_hall[selected_columns].copy()
        else:
            print(f"[CONFIG] Including ALL species abundance features ({df_hmp.shape[1]} taxa).")
            hmp_out = df_hmp.copy()
            hall_out = df_hall.copy()
    else:
        print("[CONFIG] ALL species abundance features DEACTIVATED.")
        # Create empty DataFrames keeping subject IDs as index
        hmp_out = pd.DataFrame(index=df_hmp.index)
        hall_out = pd.DataFrame(index=df_hall.index)

    # --- Load Network & Centrality Metrics ---
    hmp_m1 = pd.read_csv(METRICS_PATH / "HMP_m1_metrics.csv", index_col="subject_id")
    hall_m1 = pd.read_csv(METRICS_PATH / "Hall_m1_metrics.csv", index_col="subject_id")
    
    hmp_degbet = pd.read_csv(METRICS_PATH / "HMP_ecoli_degbet.csv", index_col="subject_id")
    hall_degbet = pd.read_csv(METRICS_PATH / "Hall_ecoli_degbet.csv", index_col="subject_id")

    hmp_ek = pd.read_csv(METRICS_PATH / "HMP_ecoli_kpneumo_weight.csv", index_col="subject_id")
    hall_ek = pd.read_csv(METRICS_PATH / "Hall_ecoli_kpneumo_weight.csv", index_col="subject_id")

    # --- Toggle: Network Density ---
    if config.get("include_density", False):
        print("[CONFIG] Attaching 'density' metric.")
        hmp_out['density'] = hmp_m1['density'].reindex(hmp_out.index).fillna(0)
        hall_out['density'] = hall_m1['density'].reindex(hall_out.index).fillna(0)

    # --- Toggle: Positive Edge Proportion ---
    if config.get("include_pos_edge_prop", False):
        print("[CONFIG] Attaching 'pos_edge_prop' metric.")
        hmp_out['pos_edge_prop'] = hmp_m1['pos_edge_prop'].reindex(hmp_out.index).fillna(0)
        hall_out['pos_edge_prop'] = hall_m1['pos_edge_prop'].reindex(hall_out.index).fillna(0)

    # --- Toggle: Mean weight ---
    if config.get("include_mean_strength", False):
        print("[CONFIG] Attaching 'mean_weight' metric.")
        hmp_out['mean_weight'] = hmp_m1['mean_weight'].reindex(hmp_out.index).fillna(0)
        hall_out['mean_weight'] = hall_m1['mean_weight'].reindex(hall_out.index).fillna(0)

    # --- Toggle: Transitivity ---
    if config.get("include_transitivity", False):
        print("[CONFIG] Attaching 'transitivity' metric.")
        hmp_out['transitivity'] = hmp_m1['transitivity'].reindex(hmp_out.index).fillna(0)
        hall_out['transitivity'] = hall_m1['transitivity'].reindex(hall_out.index).fillna(0)

    # --- Toggle: Normalized Degree ---
    if config.get("include_degree_norm", False):
        print("[CONFIG] Attaching 'degree_norm' metric.")
        hmp_out['degree_norm'] = hmp_degbet['degree_norm'].reindex(hmp_out.index).fillna(0)
        hall_out['degree_norm'] = hall_degbet['degree_norm'].reindex(hall_out.index).fillna(0)

    # --- Toggle: Normalized Betweenness ---
    if config.get("include_betweenness_norm", False):
        print("[CONFIG] Attaching 'betweenness_norm' metric.")
        hmp_out['betweenness_norm'] = hmp_degbet['betweenness_norm'].reindex(hmp_out.index).fillna(0)
        hall_out['betweenness_norm'] = hall_degbet['betweenness_norm'].reindex(hall_out.index).fillna(0)

    # --- Toggle: Absolute E. coli - K. pneumoniae Edge Weight ---
    if config.get("include_ecoli_kpneumo_w", False):
        print("[CONFIG] Attaching 'ecoli_kpneumo_abs_weight' feature.")
        hmp_out['ecoli_kpneumo_abs_weight'] = hmp_ek['ecoli_kpneumo_abs_weight'].reindex(hmp_out.index).fillna(0)
        hall_out['ecoli_kpneumo_abs_weight'] = hall_ek['ecoli_kpneumo_abs_weight'].reindex(hall_out.index).fillna(0)

    # Safety assertion to prevent training on empty feature space
    if hmp_out.shape[1] == 0:
        raise ValueError("Error: Zero features selected! Please enable at least one abundance or graph metric in FEATURE_CONFIG.")

    return hmp_out, hall_out

# Assemble feature matrices according to config
summary_counts_species, summary_counts_species_Hall = assemble_feature_matrix(
    summary_counts_species, summary_counts_species_Hall, FEATURE_CONFIG
)

print(f"\nFinal Training Features ({summary_counts_species.shape[1]} total):")
print(list(summary_counts_species.columns))


# Prepare labels for binary classification
y_multi = composite_all.groupby("subject_id")["condition"].first()
y_binary = y_multi.replace({"CD": "IBD", "UC": "IBD"})
label_mapping_binary = {label: idx for idx, label in enumerate(y_binary.unique())}
y_binary_int = y_binary.map(label_mapping_binary).loc[summary_counts_species.index]

y_multi_Hall = composite_all_Hall.groupby("subject_id")["condition"].first()
y_binary_Hall = y_multi_Hall.replace({"CD": "IBD", "UC": "IBD"})
y_binary_Hall_int = y_binary_Hall.map(label_mapping_binary).loc[summary_counts_species_Hall.index]


# Random Forest Pipeline and Hyperparameter Grid
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=11))
])

param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__class_weight': ['balanced']
}


# MNonte Carlo evaluation
NUM_REPEATS = 50
TEST_FRACTION = 0.2
INNER_FOLDS = 3

def tuned_monte_carlo(
    X, y, pipeline, param_grid,
    X_ext=None, y_ext=None,
    repeats=NUM_REPEATS,
    test_fraction=TEST_FRACTION,
    seed=200,
    scoring="average_precision"
):
    all_repeats_results = []

    for r in range(repeats):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_fraction, stratify=y, random_state=seed + r
        )

        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed + r)
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True
        )
        grid_search.fit(X_train_val, y_train_val)

        best_model = grid_search.best_estimator_

        def get_metrics(X_eval, y_eval):
            y_p = best_model.predict(X_eval)
            y_prob = best_model.predict_proba(X_eval)
            return {
                'accuracy': accuracy_score(y_eval, y_p),
                'balanced_accuracy': balanced_accuracy_score(y_eval, y_p),
                'precision': precision_score(y_eval, y_p, average='binary', zero_division=0),
                'recall': recall_score(y_eval, y_p, average='binary', zero_division=0),
                'f1_score': f1_score(y_eval, y_p, average='binary', zero_division=0),
                'auc': roc_auc_score(y_eval, y_prob[:, 1]),
                'auprc': average_precision_score(y_eval, y_prob[:, 1])
            }

        metrics_internal = get_metrics(X_test, y_test)
        metrics_external = get_metrics(X_ext, y_ext) if X_ext is not None else None

        y_scores = best_model.predict_proba(X_test)[:, 1]
        fpr_micro, tpr_micro, _ = roc_curve(y_test, y_scores)
        prec_micro, rec_micro, _ = precision_recall_curve(y_test, y_scores)

        clf = best_model.named_steps['classifier']
        importance_values = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else None

        all_repeats_results.append({
            'run': r,
            'metrics': metrics_internal,
            'metrics_external': metrics_external,
            'best_params': grid_search.best_params_,
            'selected_features': X.columns.tolist(),
            'feature_importances': importance_values,
            'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro,
            'prec_micro': prec_micro, 'rec_micro': rec_micro
        })

        print(f"[Run {r+1}/{repeats}] Internal AUC: {metrics_internal['auc']:.4f}, External AUC: {metrics_external['auc']:.4f}" if metrics_external else f"[Run {r+1}/{repeats}] Internal AUC: {metrics_internal['auc']:.4f}")

    return all_repeats_results


# Train models and plot results
print("\n--- Training Random Forest Model ---")
rf_results = tuned_monte_carlo(
    summary_counts_species, y_binary_int,
    pipeline_rf, param_grid_rf,
    X_ext=summary_counts_species_Hall, y_ext=y_binary_Hall_int,
    repeats=NUM_REPEATS,
    test_fraction=TEST_FRACTION,
    seed=11,
    scoring="average_precision"
)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

internal_aucs = [res['metrics']['auc'] for res in rf_results]
external_aucs = [res['metrics_external']['auc'] for res in rf_results]
internal_auprcs = [res['metrics']['auprc'] for res in rf_results]
external_auprcs = [res['metrics_external']['auprc'] for res in rf_results]
internal_f1s = [res['metrics']['f1_score'] for res in rf_results]
external_f1s = [res['metrics_external']['f1_score'] for res in rf_results]

# --- Subplot 1: ROC Curves ---
for res in rf_results:
    axes[0].plot(res['fpr_micro'], res['tpr_micro'], color='tab:blue', alpha=0.08, linewidth=0.8)

axes[0].plot([0, 1], [0, 1], color='grey', linestyle='--', label='Chance (AUC = 0.50)')
axes[0].plot([], [], color='tab:blue', label=f'Internal Test (AUC = {np.mean(internal_aucs):.2f} ± {np.std(internal_aucs):.2f})')
axes[0].plot([], [], color='tab:orange', label=f'External Hall (AUC = {np.mean(external_aucs):.2f} ± {np.std(external_aucs):.2f})')
axes[0].set_title("ROC Curves", fontsize=12, fontweight='bold')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[0].grid(True, linestyle=":", alpha=0.6)

# --- Subplot 2: Precision-Recall Curves ---
for res in rf_results:
    axes[1].plot(res['rec_micro'], res['prec_micro'], color='tab:blue', alpha=0.08, linewidth=0.8)

axes[1].plot([], [], color='tab:blue', label=f'Internal Test (AUPRC = {np.mean(internal_auprcs):.2f} ± {np.std(internal_auprcs):.2f})')
axes[1].plot([], [], color='tab:orange', label=f'External Hall (AUPRC = {np.mean(external_auprcs):.2f} ± {np.std(external_auprcs):.2f})')
axes[1].set_title("Precision-Recall Curves", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="lower left")
axes[1].grid(True, linestyle=":", alpha=0.6)

# --- Subplot 3: Metric Distributions ---
df_metrics = pd.DataFrame({
    'AUC': internal_aucs + external_aucs,
    'AUPRC': internal_auprcs + external_auprcs,
    'F1 Score': internal_f1s + external_f1s,
    'Evaluation': ['Internal Test'] * NUM_REPEATS + ['External Hall'] * NUM_REPEATS
})

df_melted = df_metrics.melt(id_vars=['Evaluation'], var_name='Metric', value_name='Score')

sns.boxplot(
    data=df_melted, x='Metric', y='Score', hue='Evaluation',
    ax=axes[2], palette=['tab:blue', 'tab:orange'], boxprops=dict(alpha=0.7), showmeans=True
)
sns.stripplot(
    data=df_melted, x='Metric', y='Score', hue='Evaluation',
    ax=axes[2], palette=['tab:blue', 'tab:orange'], dodge=True, alpha=0.4, jitter=0.2, legend=False
)

axes[2].set_title("Metric Distributions across Repeats", fontsize=12, fontweight='bold')
axes[2].set_xlabel("")
axes[2].set_ylabel("Score")
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.show()