import pandas as pd
import numpy as np
import copy
import multiprocessing
import pickle
from pathlib import Path
from scipy.stats import gmean
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score)

PROJECT_ROOT = Path.cwd()
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/IBD_CuratedMetagenomicData"
RESULTS_PATH = PROJECT_ROOT / "results/machine_learning/traditional_models"

# Load data
# ------------------------------------
# Load data for all timepoints
composite_all = pd.read_csv(
    PROCESSED_DATA_PATH / "HMP_counts_filtered_composite.csv", index_col="id")

# Extract taxonomic counts
counts_species = composite_all[[
    column for column in composite_all.columns if column.startswith("k__")]]

subject_ids = composite_all["subject_id"]

# Load external validation data
composite_all_Hall = pd.read_csv(
    PROCESSED_DATA_PATH / "Hall_counts_filtered_composite.csv", index_col="id")

counts_species_Hall = composite_all_Hall[[
    column for column in composite_all_Hall.columns if column.startswith("k__")]]

# Filter to only taxa present in the training data
common_taxa = counts_species.columns.intersection(counts_species_Hall.columns)
counts_species_Hall = counts_species_Hall[common_taxa]

# Insert missing taxa as zero counts in the Hall dataset
missing_taxa = counts_species.columns.difference(counts_species_Hall.columns)
for taxon in missing_taxa:
    counts_species_Hall[taxon] = 0

# Put columnns of Hall dataset in the same order as training dataset
counts_species_Hall = counts_species_Hall[counts_species.columns]

print(f"Training data shape: {counts_species.shape}, External validation data shape: {counts_species_Hall.shape}")

subject_ids_Hall = composite_all_Hall["subject_id"]


# Pre-processing functions
# ----------------------------------
# Function to aggregate taxonomic ranks
def aggregate_counts_by_taxonomic_level(counts_species, levels=("genus", "family", "order", "class", "phylum")):
    agg_tables = {}

    # Genus
    if "genus" in levels:
        genus = counts_species.copy()
        genus.columns = [
            col.rsplit("|s__", 1)[0] if col.startswith("k__") else col for col in genus.columns
        ]
        genus = genus.T.groupby(level=0).sum().T
        agg_tables["genus"] = genus

    # Family
    if "family" in levels:
        family = counts_species.copy()
        family.columns = [
            col.rsplit("|g__", 1)[0] if col.startswith("k__") else col for col in family.columns
        ]
        family = family.T.groupby(level=0).sum().T
        agg_tables["family"] = family

    # Order
    if "order" in levels:
        order = counts_species.copy()
        order.columns = [
            col.rsplit("|f__", 1)[0] if col.startswith("k__") else col for col in order.columns
        ]
        order = order.T.groupby(level=0).sum().T
        agg_tables["order"] = order

    # Class
    if "class" in levels:
        class_ = counts_species.copy()
        class_.columns = [
            col.rsplit("|o__", 1)[0] if col.startswith("k__") else col for col in class_.columns
        ]
        class_ = class_.T.groupby(level=0).sum().T
        agg_tables["class"] = class_

    # Phylum
    if "phylum" in levels:
        phylum = counts_species.copy()
        phylum.columns = [
            col.rsplit("|c__", 1)[0] if col.startswith("k__") else col for col in phylum.columns
        ]
        phylum = phylum.T.groupby(level=0).sum().T
        agg_tables["phylum"] = phylum

    return agg_tables

# Function to transform counts to clr
def clr(df, pseudocount=1):
    df = df + pseudocount  # avoid log(0)
    gm = gmean(df, axis=1)
    clr_df = np.log(df.div(gm, axis=0))
    return clr_df

# Function to drop unclassified taxa
def drop_unclassified(df, rank_prefix):
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


# Aggregate counts, CLR transform and summarize timepoints
# ------------------------------------
# Generate tables at all levels
agg_tables = aggregate_counts_by_taxonomic_level(counts_species)
agg_tables_Hall = aggregate_counts_by_taxonomic_level(counts_species_Hall)
# counts_genus = agg_tables["genus"]
# counts_family = agg_tables["family"]
# counts_order = agg_tables["order"]
# counts_class = agg_tables["class"]
# counts_phylum = agg_tables["phylum"]



# Compute CLR
clr_counts_species = clr(counts_species)
summary_counts_species = clr_counts_species.groupby(subject_ids).mean()

clr_counts_species_Hall = clr(counts_species_Hall)
summary_counts_species_Hall = clr_counts_species_Hall.groupby(subject_ids_Hall).mean()

# Filter out subjects with < 7 samples (timepoints) to match GNN data
# ------------------------------------
subject_sample_counts = subject_ids.value_counts()
valid_subjects = subject_sample_counts[subject_sample_counts >= 7].index
summary_counts_species = summary_counts_species.loc[valid_subjects]

subject_sample_counts_Hall = subject_ids_Hall.value_counts()
valid_subjects_Hall = subject_sample_counts_Hall[subject_sample_counts_Hall >= 7].index
summary_counts_species_Hall = summary_counts_species_Hall.loc[valid_subjects_Hall]

# Remove unclassified taxa in each rank
# -------------------------------------
summary_counts_species = drop_unclassified(summary_counts_species, 's__')
summary_counts_species_Hall = drop_unclassified(summary_counts_species_Hall, 's__')

# Prepare labels
# --------------------------------
# Extract labels
y_multi = composite_all.groupby("subject_id")["condition"].first()
y_binary = y_multi.replace({"CD": "IBD", "UC": "IBD"})

# Map labels to integers
label_mapping_multi = {label: idx for idx,
                       label in enumerate(y_multi.unique())}
y_multi_int = y_multi.map(label_mapping_multi)
label_mapping_binary = {label: idx for idx,
                        label in enumerate(y_binary.unique())}
y_binary_int = y_binary.map(label_mapping_binary)

# Match labels to subjects in summary_counts
# CHECKED: IBD:0, healthy:1; CD:0, UC:1, healthy:2
y_multi_int = y_multi_int.loc[summary_counts_species.index]
y_binary_int = y_binary_int.loc[summary_counts_species.index]

# Extract labels for Hall dataset
y_multi_Hall = composite_all_Hall.groupby("subject_id")["condition"].first()
y_binary_Hall = y_multi_Hall.replace({"CD": "IBD", "UC": "IBD"})
y_multi_Hall_int = y_multi_Hall.map(label_mapping_multi)
y_binary_Hall_int = y_binary_Hall.map(label_mapping_binary)

y_multi_Hall_int = y_multi_Hall_int.loc[summary_counts_species_Hall.index]
y_binary_Hall_int = y_binary_Hall_int.loc[summary_counts_species_Hall.index]

# Define models' grids
# --------------------------------
# Random Forest
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__class_weight': ['balanced']
}

# Logistic Regression
param_grid_lr = {
    'classifier__penalty': ['l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__solver': ['saga', "lbfgs"],
    'classifier__class_weight': ['balanced']
}

# K-Nearest Neighbors
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9, 12],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

# Support Vector Machine
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__class_weight': ['balanced']
}

# Multi-Layer Perceptron
param_grid_mlp = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'classifier__activation': ['relu'],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__learning_rate_init': [0.001]
}

# Define pipelines
# --------------------------------
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    # remove features with zero variance
    ('variance_filter', VarianceThreshold(threshold=0.0)),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', RandomForestClassifier(random_state=11))
])
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_filter', VarianceThreshold(threshold=0.0)),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', LogisticRegression(random_state=11, max_iter=5000))
])
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_filter', VarianceThreshold(threshold=0.0)),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', KNeighborsClassifier())
])
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_filter', VarianceThreshold(threshold=0.0)),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', SVC(random_state=11, probability=True))
])
pipeline_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_filter', VarianceThreshold(threshold=0.0)),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', MLPClassifier(random_state=11, max_iter=1500))
])

# Function to perform tuned Monte Carlo.
# --------------------------------
NUM_REPEATS = 50
TEST_FRACTION = 0.2
INNER_FOLDS = 3

def tuned_monte_carlo(
        X,
        y,
        pipeline,
        param_grid,
        X_ext=None,
        y_ext=None,
        repeats=NUM_REPEATS,
        test_fraction=TEST_FRACTION,
        seed=11,
        scoring="roc_auc"
):
    all_repeats_results = []

    for r in range(repeats):
        # Split data into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_fraction, stratify=y, random_state=seed + r)

        print(f"\nRepeat {r + 1}/{repeats}: Test set size = {len(X_test)}")

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=seed + r)

        # Compute the number of features and k_values for feature selection
        n_features = X_train_val.shape[1]
        k_values = [int(n_features * frac)
                    for frac in [0.2, 0.5, 0.8, 1.0] if int(n_features * frac) > 1]
        if len(k_values) < 2:
            k_values = [max(1, int(n_features * 0.5)), n_features]

        # Make a copy of param_grid and set the feature selection parameter
        grid = copy.deepcopy(param_grid)
        grid["feature_selection__k"] = k_values

        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True
        )
        grid_search.fit(X_train_val, y_train_val)

        # Best model from inner CV
        best_model = grid_search.best_estimator_

        def get_metrics(X_eval, y_eval):
            y_p = best_model.predict(X_eval)
            y_prob = best_model.predict_proba(X_eval)

            if len(np.unique(y)) == 2:
                return {
                    'accuracy': accuracy_score(y_eval, y_p),
                    'balanced_accuracy': balanced_accuracy_score(y_eval, y_p),
                    'precision': precision_score(y_eval, y_p, average='binary', zero_division=0),
                    'recall': recall_score(y_eval, y_p, average='binary', zero_division=0),
                    'f1_score': f1_score(y_eval, y_p, average='binary', zero_division=0),
                    'auc': roc_auc_score(y_eval, y_prob[:, 1]),
                    'auprc': average_precision_score(y_eval, y_prob[:, 1])
                }
            elif len(np.unique(y)) > 2:
                return {
                    'accuracy': accuracy_score(y_eval, y_p),
                    'balanced_accuracy': balanced_accuracy_score(y_eval, y_p),
                    'precision_macro': precision_score(y_eval, y_p, average='macro', zero_division=0),
                    'precision_micro': precision_score(y_eval, y_p, average='micro', zero_division=0),
                    'recall_macro': recall_score(y_eval, y_p, average='macro', zero_division=0),
                    'recall_micro': recall_score(y_eval, y_p, average='micro', zero_division=0),
                    'f1_score_macro': f1_score(y_eval, y_p, average='macro', zero_division=0),
                    'f1_score_micro': f1_score(y_eval, y_p, average='micro', zero_division=0),
                    'auc_macro': roc_auc_score(y_eval, y_prob, average="macro", multi_class='ovr'),
                    'auc_micro': roc_auc_score(y_eval, y_prob, average="micro", multi_class='ovr'),
                    'auprc_macro': average_precision_score(y_eval, y_prob, average='macro'),
                    'auprc_micro': average_precision_score(y_eval, y_prob, average='micro')
                }
            else:
                raise ValueError("y must have at least two classes.")

        # Collect metrics for internal and external validation
        metrics_internal = get_metrics(X_test, y_test)
        y_pred = best_model.predict(X_test)

        if X_ext is not None and y_ext is not None:
            metrics_external = get_metrics(X_ext, y_ext)
            y_pred_ext = best_model.predict(X_ext)
        else:
            metrics_external = None
            y_pred_ext = None

        # Initialize curve variables to store Micro and Macro data
        fpr_micro, tpr_micro, prec_micro, rec_micro = None, None, None, None
        fpr_macro, tpr_macro, prec_macro, rec_macro = None, None, None, None

        if hasattr(best_model, "predict_proba"):
            y_scores = best_model.predict_proba(X_test)
            classes = np.unique(y_test)
            n_classes = len(classes)

            if n_classes == 2:
                # --- Binary Case ---
                probs = y_scores[:, 1] if y_scores.ndim > 1 else y_scores

                # Micro curves are calculated as the standard binary curves
                fpr_micro, tpr_micro, _ = roc_curve(y_test, probs)
                prec_micro, rec_micro, _ = precision_recall_curve(
                    y_test, probs)

                # Macro curves remain None (or can be set equal to Micro)
                fpr_macro, tpr_macro, prec_macro, rec_macro = None, None, None, None

            else:
                # --- Multiclass Case (n_classes > 2) ---

                # 1. Prepare data for OvR
                y_test_bin = label_binarize(y_test, classes=classes)

                # --- A. Micro-Averaged Curves (Simple Flattening) ---

                # ROC Micro
                fpr_micro, tpr_micro, _ = roc_curve(
                    y_test_bin.ravel(), y_scores.ravel())

                # PR Micro
                prec_micro, rec_micro, _ = precision_recall_curve(
                    y_test_bin.ravel(), y_scores.ravel())

                # --- B. Macro-Averaged Curves (OvR, Interpolation, and Averaging) ---

                # Store individual ROC and PR curve components
                fpr = dict()
                tpr = dict()
                prec = dict()
                rec = dict()

                for i in range(n_classes):
                    # ROC Curve for class i vs. all others
                    fpr[i], tpr[i], _ = roc_curve(
                        y_test_bin[:, i], y_scores[:, i])

                    # PR Curve for class i vs. all others
                    prec[i], rec[i], _ = precision_recall_curve(
                        y_test_bin[:, i], y_scores[:, i])

                # --- Macro ROC Interpolation ---

                # 1. Aggregate all FPRs into a single array for common interpolation points
                all_fpr = np.unique(np.concatenate(
                    [fpr[i] for i in range(n_classes)]))

                # 2. Interpolate all ROC curves at these common points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                # 3. Average and set Macro ROC results
                mean_tpr /= n_classes
                fpr_macro = all_fpr
                tpr_macro = mean_tpr

                # --- Macro PR Interpolation ---

                # 1. Define a common set of Recall points (e.g., 0 to 1 in 100 steps)
                all_rec = np.unique(np.concatenate(
                    [rec[i] for i in range(n_classes)]))

                def safe_interp(x, xp, yp):
                    xp_unique, indices = np.unique(xp, return_index=True)
                    yp_unique = yp[indices]
                    return np.interp(x, xp_unique, yp_unique)

                # 2. Interpolate and average the Precision values
                mean_prec = np.zeros_like(all_rec)
                for i in range(n_classes):
                    # PR curves are usually defined from high recall (low threshold) to low recall (high threshold).
                    # Reverse the arrays for np.interp, which expects increasing x (recall) values.
                    mean_prec += safe_interp(all_rec,
                                             rec[i][::-1], prec[i][::-1])

                # 3. Average and set Macro PR results
                mean_prec /= n_classes
                prec_macro = mean_prec
                rec_macro = all_rec

        # Selected features
        selected_features = X_train_val.columns[best_model.named_steps['feature_selection'].get_support(
        )]

        # Sample tracking
        sample_tracking_internal = pd.DataFrame({
            'sample': X_test.index,
            'true_label': y_test,
            'predicted_label': y_pred,
            'correct': y_test.values == y_pred
        })

        if X_ext is not None and y_ext is not None:
            sample_tracking_external = pd.DataFrame({
                'sample': X_ext.index,
                'true_label': y_ext,
                'predicted_label': y_pred_ext,
                'correct': y_ext.values == y_pred_ext
            })
        else:
            sample_tracking_external = None

        clf = best_model.named_steps['classifier']

        # Extract either RF importances or LR coefficients
        if hasattr(clf, 'feature_importances_'):
            importance_values = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            # For binary LR, coef_ is [[feat1, feat2, ...]] so take the first row
            importance_values = clf.coef_[0]
        else:
            importance_values = None

        # Collect repeat results
        repeat_results = {
            'run': r,
            'metrics': metrics_internal,
            'metrics_external': metrics_external,
            'best_params': grid_search.best_params_,
            'selected_features': selected_features.tolist(),
            'feature_importances': importance_values,
            'fpr_micro': fpr_micro, 'tpr_micro': tpr_micro,
            'prec_micro': prec_micro, 'rec_micro': rec_micro,
            'fpr_macro': fpr_macro, 'tpr_macro': tpr_macro,
            'prec_macro': prec_macro, 'rec_macro': rec_macro,
            'sample_tracking': sample_tracking_internal,
            'sample_tracking_external': sample_tracking_external,
            'train_samples': X_train_val,
            'test_samples': X_test
        }

        all_repeats_results.append(repeat_results)

    return all_repeats_results


# Organize pipelines and datasets for execution
# --------------------------------
pipelines = {
    'KNN': (pipeline_knn, param_grid_knn),
    'LR': (pipeline_lr, param_grid_lr),
    'SVM': (pipeline_svm, param_grid_svm),
    'RF': (pipeline_rf, param_grid_rf),
    'MLP': (pipeline_mlp, param_grid_mlp)
}

longitudinal_datasets_binary = {
    'species': (summary_counts_species, y_binary_int)
}

# Run models and save results
# --------------------------------
# Create results directory if it doesn't exist
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Helper to run and collect results for all models
def run_and_save_all_models(datasets, label_type, dataset_type, X_ext=None, y_ext=None,):
    for data_name, (X, y) in datasets.items():
        all_results = {}
        for model_name, (pipe, grid) in pipelines.items():
            print(
                f"\nRunning model: {model_name} on dataset: {data_name} ({label_type}, {dataset_type})")
            res = tuned_monte_carlo(
                X, y, pipe, grid,
                X_ext=X_ext, y_ext=y_ext,
                repeats=NUM_REPEATS,
                test_fraction=TEST_FRACTION,
                seed=11,
                scoring="average_precision" # Use AUPRC for tuning since it's more informative for imbalanced datasets
            )
            all_results[model_name] = res

        # Save all model results for this dataset/taxrank/label combination
        with open(RESULTS_PATH / f"{dataset_type}_{data_name}_{label_type}_MCarlo.pkl", "wb") as f:
            pickle.dump(all_results, f)


# Run for all combinations
run_and_save_all_models(longitudinal_datasets_binary, "binary", "HMP_Train_Hall_External", X_ext=summary_counts_species_Hall, y_ext=y_binary_Hall_int)
