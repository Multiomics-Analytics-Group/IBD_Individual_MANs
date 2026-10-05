import torch
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
from torch_geometric.utils import degree
from IPython.display import clear_output
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from collections import defaultdict

# --- TRAINING AND VALIDATION FUNCTIONS FOR GRAPH MODELS ----

def concat_node_features(data, feature_list, include_flags=False):
    x_parts = []

    for feat_name in feature_list:
        # append the feature itself
        x_parts.append(getattr(data, feat_name))

        if include_flags:
            # construct the reliability flag attribute name
            has_name = "has_" + feat_name.split("x_")[-1]
            if hasattr(data, has_name):
                x_parts.append(getattr(data, has_name))

    return torch.cat(x_parts, dim=-1)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, patience, scheduler=None, node_features_list = None,
                use_edge_weights=True, threshold = 0.5, y_key="y_binary", plot_losses=False,
                plot_interval=5, repeat_idx=0, include_flags=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_loss = float('inf')
    best_model_state = None
    counter = 0

    history = defaultdict(list)
    fold_title = f"Current run: Repeat {repeat_idx}"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data in train_loader:

            # 1. Prepare inputs and targets
            # node_features = getattr(data, node_features_key).to(device)
            node_features = concat_node_features(data, node_features_list, include_flags = include_flags).to(device)
            y = getattr(data, y_key)

            # Ensure y is a float tensor on the correct device
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float, device=device)
            else:
                y = y.float().to(device)

            # Move graph data to device
            data = data.to(device)

            optimizer.zero_grad()

            # 2. Forward pass (returns logits of shape [batch_size])
            out, _, _ = model(data, node_features=node_features,
                              use_edge_weights=use_edge_weights)

            # 3. Loss calculation
            loss = criterion(out, y.view(-1))

            loss.backward()
            optimizer.step()

            # Weight loss by number of graphs
            total_loss += loss.item() * data.num_graphs

        total_loss /= len(train_loader.dataset)
        history["train_loss"].append(total_loss)

        # --- Validation ---
        val_metrics, _, _, _ = validate_model(
            model, val_loader, criterion,
            # node_features_key=node_features_key,
            node_features_list=node_features_list,
            use_edge_weights=use_edge_weights,
            threshold=threshold,
            y_key=y_key,
            device=device,
            include_flags=include_flags
        )
        val_loss, val_acc, val_auc, val_auprc, val_prec, val_rec, val_f1, fpr, tpr, thresh_roc, prec, rec, thresh_prc = val_metrics

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_auprc"].append(val_auprc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["fpr"].append(fpr)
        history["tpr"].append(tpr)
        history["thresh_roc"].append(thresh_roc)
        history["prec"].append(prec)
        history["rec"].append(rec)
        history["thresh_prc"].append(thresh_prc)

        # --- SCHEDULER STEP ---
        if scheduler is not None:
            scheduler.step(val_loss)
        # -----------------------

        # --- REAL-TIME PLOTTING LOGIC ---
        if plot_losses:
            if (epoch + 1) % plot_interval == 0 or (epoch + 1) == num_epochs:
                clear_output(wait=True)

                train_losses = history['train_loss']
                val_losses = history['val_loss']

                plt.figure(figsize=(9, 5))
                epochs_ran = range(1, len(train_losses) + 1)

                plt.plot(epochs_ran, train_losses, label='Training Loss',
                         marker='o', linestyle='-', markersize=4)
                plt.plot(epochs_ran, val_losses, label='Validation Loss',
                         marker='s', linestyle='--', markersize=4)

                plt.title(f"{fold_title} Loss Curve (Epoch {epoch + 1})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, alpha=0.6)
                plt.show()
        # -------------------------------

        # --- Early stopping ---
        if val_loss < best_loss:
            best_loss = val_loss
            # Deep copy the state dict to save the best model weights
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}!")
                break

        # Optional print for epoch progress
        print(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")

    # Load the weights of the best performing model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history

# this function is the same as train_model() but includes edge weights in the forward pass
# to be used with integrated gradients
def train_model2 (model, train_loader, val_loader, criterion, optimizer,
                num_epochs, patience, scheduler=None, node_features_list = None,
                use_edge_weights=True, threshold = 0.5, y_key="y_binary", plot_losses=False,
                plot_interval=5, repeat_idx=0, include_flags=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_loss = float('inf')
    best_model_state = None
    counter = 0

    history = defaultdict(list)
    fold_title = f"Current run: Repeat {repeat_idx}"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data in train_loader:

            # 1. Prepare inputs and targets
            # node_features = getattr(data, node_features_key).to(device)
            node_features = concat_node_features(data, node_features_list, include_flags = include_flags).to(device)
            y = getattr(data, y_key)

            # Ensure y is a float tensor on the correct device
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float, device=device)
            else:
                y = y.float().to(device)

            # Move graph data to device
            data = data.to(device)

            edge_w = data.weight if use_edge_weights and hasattr(data, 'weight') else None

            optimizer.zero_grad()

            # 2. Forward pass (returns logits of shape [batch_size])
            out, _, _ = model(data, node_features=node_features,
                              use_edge_weights=use_edge_weights, edge_weights=edge_w)

            # 3. Loss calculation
            loss = criterion(out, y.view(-1))

            loss.backward()
            optimizer.step()

            # Weight loss by number of graphs
            total_loss += loss.item() * data.num_graphs

        total_loss /= len(train_loader.dataset)
        history["train_loss"].append(total_loss)

        # --- Validation ---
        val_metrics, _, _, _ = validate_model2(
            model, val_loader, criterion,
            # node_features_key=node_features_key,
            node_features_list=node_features_list,
            use_edge_weights=use_edge_weights,
            threshold=threshold,
            y_key=y_key,
            device=device,
            include_flags=include_flags
        )
        val_loss, val_acc, val_auc, val_auprc, val_prec, val_rec, val_f1, fpr, tpr, thresh_roc, prec, rec, thresh_prc = val_metrics

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_auprc"].append(val_auprc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)
        history["fpr"].append(fpr)
        history["tpr"].append(tpr)
        history["thresh_roc"].append(thresh_roc)
        history["prec"].append(prec)
        history["rec"].append(rec)
        history["thresh_prc"].append(thresh_prc)

        # --- SCHEDULER STEP ---
        if scheduler is not None:
            scheduler.step(val_loss)
        # -----------------------

        # --- REAL-TIME PLOTTING LOGIC ---
        if plot_losses:
            if (epoch + 1) % plot_interval == 0 or (epoch + 1) == num_epochs:
                clear_output(wait=True)

                train_losses = history['train_loss']
                val_losses = history['val_loss']

                plt.figure(figsize=(9, 5))
                epochs_ran = range(1, len(train_losses) + 1)

                plt.plot(epochs_ran, train_losses, label='Training Loss',
                         marker='o', linestyle='-', markersize=4)
                plt.plot(epochs_ran, val_losses, label='Validation Loss',
                         marker='s', linestyle='--', markersize=4)

                plt.title(f"{fold_title} Loss Curve (Epoch {epoch + 1})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, alpha=0.6)
                plt.show()
        # -------------------------------

        # --- Early stopping ---
        if val_loss < best_loss:
            best_loss = val_loss
            # Deep copy the state dict to save the best model weights
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}!")
                break

        # Optional print for epoch progress
        print(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")

    # Load the weights of the best performing model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, history


def validate_model(model, loader, criterion, node_features_list = None,
                   use_edge_weights=True, y_key="y_binary", threshold=0.5, device=None, include_flags=False):

    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_loss = 0
    all_probs, all_preds, all_targets = [], [], []
    is_multiclass = model.num_classes > 1

    with torch.no_grad():

        for data in loader:
            # node_features = getattr(data, node_features_key).to(device)
            node_features = concat_node_features(data, node_features_list, include_flags = include_flags).to(device)
            y = getattr(data, y_key)

            if not isinstance(y, torch.Tensor):
                y = torch.tensor(
                    y, dtype=torch.float if not is_multiclass else torch.long, device=device)
            else:
                y = y.to(device)

            data = data.to(device)

            # Forward pass: out is logits
            out, _, _ = model(data, node_features=node_features,
                              use_edge_weights=use_edge_weights)

            # Loss calculation: Ensure 'out' and 'y' are compatible shapes
            loss = criterion(out, y.view(-1).float()
                             if not is_multiclass else y)
            val_loss += loss.item() * data.num_graphs

            # Predictions / probabilities
            if not is_multiclass:
                # Binary case (out is 1D logits)
                probs = torch.sigmoid(out).view(-1)
                preds = (probs >= threshold).long()
            else:
                # Multi-class case (out is [batch_size, num_classes])
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            # targets must be 1D for metrics
            all_targets.append(y.view(-1).cpu())

    # Average loss per graph
    val_loss /= len(loader.dataset)

    # Concatenate all batches for metrics
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Accuracy
    val_acc = (all_preds == all_targets).float().mean().item()

    # ROC-AUC
    try:
        if is_multiclass:
            # For multi-class, use the probabilities of all classes
            val_auc = roc_auc_score(
                all_targets.numpy(), all_probs.numpy(), multi_class='ovr')
            val_auprc = average_precision_score(
                all_targets.numpy(), all_probs.numpy(), average='macro')
        else:
            fpr, tpr, thresh_roc, prec, rec, thresh_prc = None, None, None, None, None, None
            # For binary, use the probability of the positive class (all_probs is 1D)
            val_auc = roc_auc_score(all_targets.numpy(), all_probs.numpy())
            fpr, tpr, thresh_roc = roc_curve(
                all_targets.numpy(), all_probs.numpy())
            val_auprc = average_precision_score(
                all_targets.numpy(), all_probs.numpy())
            prec, rec, thresh_prc = precision_recall_curve(
                all_targets.numpy(), all_probs.numpy())

    except ValueError:
        # Handles cases where there's only one class present in the batch
        val_auc = float('nan')
        val_auprc = float('nan')

    # Precision, Recall, F1 score
    average_type = 'binary' if not is_multiclass else 'macro'
    val_prec = precision_score(
        all_targets.numpy(), all_preds.numpy(), average=average_type, zero_division=0)
    val_rec = recall_score(
        all_targets.numpy(), all_preds.numpy(), average=average_type, zero_division=0)
    val_f1 = f1_score(all_targets.numpy(), all_preds.numpy(),
                      average=average_type)

    metrics = (val_loss, val_acc, val_auc,
               val_auprc, val_prec, val_rec, val_f1,
               fpr, tpr, thresh_roc,
               prec, rec, thresh_prc)

    return metrics, all_probs, all_preds, all_targets

# This function is the same as validate_model() but includes edge weights in the forward pass
# to be used with integrated gradients
def validate_model2(model, loader, criterion, node_features_list = None,
                   use_edge_weights=True, y_key="y_binary", threshold=0.5, device=None, include_flags=False):

    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    val_loss = 0
    all_probs, all_preds, all_targets = [], [], []
    is_multiclass = model.num_classes > 1

    with torch.no_grad():

        for data in loader:
            # node_features = getattr(data, node_features_key).to(device)
            node_features = concat_node_features(data, node_features_list, include_flags = include_flags).to(device)
            y = getattr(data, y_key)

            if not isinstance(y, torch.Tensor):
                y = torch.tensor(
                    y, dtype=torch.float if not is_multiclass else torch.long, device=device)
            else:
                y = y.to(device)

            data = data.to(device)

            edge_w = data.weight if use_edge_weights and hasattr(data, 'weight') else None

            # Forward pass: out is logits
            out, _, _ = model(data, node_features=node_features,
                              use_edge_weights=use_edge_weights, edge_weights=edge_w)

            # Loss calculation: Ensure 'out' and 'y' are compatible shapes
            loss = criterion(out, y.view(-1).float()
                             if not is_multiclass else y)
            val_loss += loss.item() * data.num_graphs

            # Predictions / probabilities
            if not is_multiclass:
                # Binary case (out is 1D logits)
                probs = torch.sigmoid(out).view(-1)
                preds = (probs >= threshold).long()
            else:
                # Multi-class case (out is [batch_size, num_classes])
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            # targets must be 1D for metrics
            all_targets.append(y.view(-1).cpu())

    # Average loss per graph
    val_loss /= len(loader.dataset)

    # Concatenate all batches for metrics
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Accuracy
    val_acc = (all_preds == all_targets).float().mean().item()

    # ROC-AUC
    try:
        if is_multiclass:
            # For multi-class, use the probabilities of all classes
            val_auc = roc_auc_score(
                all_targets.numpy(), all_probs.numpy(), multi_class='ovr')
            val_auprc = average_precision_score(
                all_targets.numpy(), all_probs.numpy(), average='macro')
        else:
            # For binary, use the probability of the positive class (all_probs is 1D)
            val_auc = roc_auc_score(all_targets.numpy(), all_probs.numpy())
            fpr, tpr, thresh_roc = roc_curve(
                all_targets.numpy(), all_probs.numpy())
            val_auprc = average_precision_score(
                all_targets.numpy(), all_probs.numpy())
            prec, rec, thresh_prc = precision_recall_curve(
                all_targets.numpy(), all_probs.numpy())

    except ValueError:
        # Handles cases where there's only one class present in the batch
        val_auc = float('nan')
        val_auprc = float('nan')

    # Precision, Recall, F1 score
    average_type = 'binary' if not is_multiclass else 'macro'
    val_prec = precision_score(
        all_targets.numpy(), all_preds.numpy(), average=average_type, zero_division=0)
    val_rec = recall_score(
        all_targets.numpy(), all_preds.numpy(), average=average_type, zero_division=0)
    val_f1 = f1_score(all_targets.numpy(), all_preds.numpy(),
                      average=average_type)

    metrics = (val_loss, val_acc, val_auc,
               val_auprc, val_prec, val_rec, val_f1,
               fpr, tpr, thresh_roc,
               prec, rec, thresh_prc)

    return metrics, all_probs, all_preds, all_targets

#  --- EXTRACT GRAPH AND NODE EMBEDDINGS FUNCTION ----


def extract_embeddings_df(
    model,
    loader,
    # node_features_key='x_taxonomy_onehot',
    node_features_list = None,
    use_edge_weights=True,
    threshold=0.5,
    y_key="y_binary",
    device=None,
    include_flags=False
):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.eval()

    all_graph_embs, all_node_embs = [], []
    all_graph_labels, all_node_labels = [], []
    all_graph_preds, all_node_preds = [], []
    all_node_names = []
    all_node_graph_names = []
    all_graph_names = []
    is_binary = model.num_classes == 1

    with torch.no_grad():
        for data in loader:

            # node_features = getattr(data, node_features_key).to(device)
            node_features = concat_node_features(data, node_features_list, include_flags = include_flags).to(device)
            y = getattr(data, y_key)

            if not isinstance(y, torch.Tensor):
                y = torch.tensor(
                    y,
                    dtype=torch.float if is_binary else torch.long
                ).to(device)

            data = data.to(device)

            # Forward pass: gets logits, graph_emb, node_emb
            logits, graph_emb, node_emb = model(
                data, node_features=node_features, use_edge_weights=use_edge_weights
            )

            # --- Predictions ---
            if is_binary:
                probs = torch.sigmoid(logits.view(-1))
                preds = (probs >= threshold).long()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            # ---- Graph-level ----
            all_graph_embs.append(graph_emb.cpu())
            all_graph_labels.append(y.view(-1).cpu())
            # store predictions
            all_graph_preds.append(preds.view(-1).cpu())

            # Graph names
            if hasattr(data, "name"):
                if isinstance(data.name, (list, tuple)):
                    graph_names = list(data.name)
                elif isinstance(data.name, str):
                    graph_names = [data.name]
                else:
                    graph_names = [
                        f"graph_{i}" for i in range(graph_emb.shape[0])]
            else:
                graph_names = [f"graph_{i}" for i in range(graph_emb.shape[0])]
            all_graph_names.extend(graph_names)

            # ---- Node-level ----
            if node_emb is not None:

                # Node labels
                node_labels = y[data.batch].cpu()
                all_node_embs.append(node_emb.cpu())
                all_node_labels.append(node_labels)
                # store predictions
                node_predictions = preds[data.batch].cpu()
                all_node_preds.append(node_predictions)

                # Node names (flatten nested lists per graph)
                if hasattr(data, "node_name"):
                    flat_node_names = []
                    flat_node_graph_names = []
                    for i, gname in enumerate(graph_names):

                        # list of node names in graph i
                        names_in_graph = data.node_name[i]
                        flat_node_names.extend(names_in_graph)
                        flat_node_graph_names.extend(
                            [gname] * len(names_in_graph))

                    node_names = flat_node_names
                    node_graph_names = flat_node_graph_names

                    # sanity check
                    assert len(node_names) == node_emb.shape[0], \
                        f"Node name count mismatch: {len(node_names)} vs {node_emb.shape[0]}"
                    assert len(node_graph_names) == node_emb.shape[0]
                else:
                    node_names = [
                        f"node_{i}" for i in range(node_emb.shape[0])]
                    node_graph_names = [graph_names[i]
                                        for i in data.batch.cpu()]

                all_node_names.extend(node_names)
                all_node_graph_names.extend(node_graph_names)

    # --- Combine and create DataFrames ---
    # Graph-level
    graph_embs = torch.cat(all_graph_embs, dim=0)
    graph_labels = torch.cat(all_graph_labels, dim=0)
    graph_preds = torch.cat(all_graph_preds, dim=0)

    graph_records = [
        list(emb.numpy()) + [label.item(), preds.item(), name]
        for emb, label, preds, name in zip(graph_embs, graph_labels, graph_preds, all_graph_names)
    ]
    graph_col_names = [f"dim_{i}" for i in range(
        graph_embs.shape[1])] + ['label', 'preds', 'graph_name']
    graph_embeddings_df = pd.DataFrame(graph_records, columns=graph_col_names)

    # Node-level
    node_embeddings_df = None
    if all_node_embs:
        node_embs = torch.cat(all_node_embs, dim=0)
        node_labels = torch.cat(all_node_labels, dim=0)
        node_preds = torch.cat(all_node_preds, dim=0)
        node_records = [
            list(emb.numpy()) + [label.item(),
                                 node_name, preds.item(), graph_name]
            for emb, label, node_name, preds, graph_name in zip(node_embs, node_labels, all_node_names, node_preds, all_node_graph_names)
        ]
        node_col_names = [f"dim_{i}" for i in range(
            node_embs.shape[1])] + ['label', 'node_name', 'preds', 'graph_name']
        node_embeddings_df = pd.DataFrame(node_records, columns=node_col_names)

    return graph_embeddings_df, node_embeddings_df


def get_graph_names_and_labels(dataset, indices):
    records = []
    for idx in indices:
        # The Subset object's index is relative to its original dataset.
        # We access the original Data object via the Subset's __getitem__.
        data_obj = dataset[idx]
        graph_name = getattr(data_obj, 'name', f'graph_{idx}')
        true_label = data_obj.y_binary.item() if hasattr(data_obj, 'y_binary') else np.nan
        records.append({'graph_name': graph_name, 'true_label': true_label})
    return pd.DataFrame(records)


# --- GENERAL UTILITIES FOR GRAPH MODELS ----


def set_seed(seed):

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

        # Further CUDA settings for determinism
        # This can incur a performance hit, but guarantees reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Environment variables (useful for certain frameworks/libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)


def collect_dataset_stats_by_label(pyg_dataset):

    # Organize data by label
    labeled_data = defaultdict(lambda: defaultdict(list))

    for data in pyg_dataset:
        label = data.y_binary.item()  # Assuming y_binary holds the label (0 or 1)

        N = data.num_nodes
        E = data.num_edges / 2  # Unique edges

        if N > 0:
            labeled_data[label]['num_nodes'].append(N)
            labeled_data[label]['num_edges'].append(int(E))

            if N > 1:
                # Degree
                row, col = data.edge_index
                deg = degree(row, N).tolist()
                labeled_data[label]['degree'].extend(deg)

                # Density: E / (N * (N - 1) / 2)
                density = E / (N * (N - 1) / 2)
                labeled_data[label]['density'].append(density)
            else:
                # Density is 0 for isolated nodes or single node graphs
                labeled_data[label]['density'].append(0.0)

            # Edge Weight
            if hasattr(data, 'weight') and data.weight is not None:
                # Only take unique edge weights (half the size)
                labeled_data[label]['edge_weight'].extend(
                    data.weight[::2].abs().tolist())

    # Calculate final statistics for each label
    final_stats = {}

    for label in [0, 1]:
        prefix = f"label_{label}_"
        data_lists = labeled_data[label]
        num_graphs = len(data_lists['num_nodes'])

        if num_graphs == 0:
            final_stats[prefix + 'num_graphs'] = 0
            continue

        final_stats[prefix + 'num_graphs'] = num_graphs

        # --- Node/Edge Counts and Density ---
        for metric in ['num_nodes', 'num_edges', 'density']:
            values = data_lists[metric]
            if values:
                final_stats[prefix + f'min_{metric}'] = np.min(values)
                final_stats[prefix + f'max_{metric}'] = np.max(values)
                final_stats[prefix + f'avg_{metric}'] = np.mean(values)

        # --- Aggregated Node Properties (Degree) ---
        if data_lists['degree']:
            final_stats[prefix +
                        'min_node_degree'] = np.min(data_lists['degree'])
            final_stats[prefix +
                        'max_node_degree'] = np.max(data_lists['degree'])
            final_stats[prefix +
                        'avg_node_degree'] = np.mean(data_lists['degree'])

        # --- Aggregated Edge Properties (Weight) ---
        if data_lists['edge_weight']:
            final_stats[prefix +
                        'min_edge_weight'] = np.min(data_lists['edge_weight'])
            final_stats[prefix +
                        'max_edge_weight'] = np.max(data_lists['edge_weight'])
            final_stats[prefix +
                        'avg_edge_weight'] = np.mean(data_lists['edge_weight'])

    return final_stats