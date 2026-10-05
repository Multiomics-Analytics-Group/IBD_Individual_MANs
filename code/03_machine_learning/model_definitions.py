import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops
from torch_geometric.nn import GraphConv, global_add_pool, global_mean_pool, SAGPooling, global_max_pool, GraphNorm, SAGEConv, GCNConv, GATv2Conv
from torch_geometric.nn.dense import dense_diff_pool


# -- MODEL CLASSES --

# Flexible GNN class
class FlexibleGNN(torch.nn.Module):
    def __init__(self, model_type, num_layers,
                 input_dim, hidden_dim, embedding_dim,
                 dropout, num_classes=1, pooling_method="add",
                 difpool_clusters_ratio=0.3, sagpool_ratio=0.6,
                 gat_heads=4):

        super(FlexibleGNN, self).__init__()

        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_method = pooling_method
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.difpool_clusters_ratio = difpool_clusters_ratio
        self.sagpool_ratio = sagpool_ratio
        self.gat_heads = gat_heads

        #  GNN layers 
        self.convs = torch.nn.ModuleList()
        conv_layer = {
            'GraphConv': GraphConv,
            'SAGEConv': SAGEConv,
            'GCNConv': GCNConv,
            'GATConv': GATv2Conv
        }.get(model_type, None)
        if conv_layer is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        # GATConv output dimension is `out_channels * heads` if concat=True (default)
        if model_type == 'GATConv':
            def intermediate_conv(in_d, out_d): return conv_layer(
                in_d, out_d // gat_heads, heads=gat_heads, dropout=dropout)

            def final_conv(in_d, out_d): return conv_layer(
                in_d, out_d, heads=1, concat=False, dropout=dropout)
        else:
            intermediate_conv = conv_layer
            final_conv = conv_layer

        # First layer: input_dim -> hidden_dim
        self.convs.append(intermediate_conv(input_dim, hidden_dim))

        # Intermediate layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(intermediate_conv(hidden_dim, hidden_dim))

        # Last layer: hidden_dim -> embedding_dim
        # Note: GATConv must typically set concat=False (or heads=1) in the final layer
        # if the desired output dimension is exactly `embedding_dim` and not
        # `embedding_dim * heads`. A custom final_conv function is used.
        if model_type == 'GATConv':
            self.convs.append(final_conv(hidden_dim, embedding_dim))
        else:
            self.convs.append(final_conv(hidden_dim, embedding_dim))

        #  GraphNorm layers 
        # Note: GATConv output dim is `out_channels * heads`, but we designed
        # the previous block so the output dim matches `hidden_dim`
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(GraphNorm(hidden_dim))

        #  DiffPool layers 
        if pooling_method == "diffpool":
            # Fixing the assignment layers to use GraphConv
            self.diffpool_assign_1 = GraphConv(embedding_dim, embedding_dim)

            num_clusters = int(difpool_clusters_ratio * embedding_dim)
            self.diffpool_assign_2 = GraphConv(
                embedding_dim, num_clusters)

        #  SAGPool layers 
        if pooling_method == "sagpool":
            self.sagpool = SAGPooling(embedding_dim, ratio=sagpool_ratio)

        #  Prediction head 
        self.pred_head = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, data, node_features=None, use_edge_weights=False, add_loops=True):
        x = node_features if node_features is not None else data.x_taxonomy_onehot
        edge_index, batch = data.edge_index, data.batch
        num_nodes = data.num_nodes

        if use_edge_weights and hasattr(data, "weight") and data.weight is not None:
            edge_weight = data.weight
            if add_loops:
                edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes)
        else:
            if add_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            edge_weight = None


        for i in range(self.num_layers):
            if edge_weight is not None and self.model_type not in ["SAGEConv", "GATConv"]:
                x = self.convs[i](x, edge_index, edge_weight=abs(edge_weight)) # Use Abs
            else:
                x = self.convs[i](x, edge_index)

            if i != self.num_layers - 1:
                x = self.bns[i](x, batch=batch)
                x = torch.relu(x)
                x = torch.nn.functional.dropout(
                    x, p=self.dropout, training=self.training)

        #  Node embeddings 
        node_embeddings = x.clone()  # [num_nodes, embedding_dim]

        #  Graph embeddings 
        if self.pooling_method == "add":
            graph_embeddings = global_add_pool(x, batch)
        elif self.pooling_method == "mean":
            graph_embeddings = global_mean_pool(x, batch)
        elif self.pooling_method == "max":
            graph_embeddings = global_max_pool(x, batch)
        elif self.pooling_method == "diffpool":

            # compute dense versions
            x_dense, mask = to_dense_batch(x, batch)
            adj_dense = to_dense_adj(edge_index, batch=batch)

            # Compute assignment matrix S
            S = F.relu(self.diffpool_assign_1(x, edge_index))
            S = self.diffpool_assign_2(S, edge_index)
            S_dense, _ = to_dense_batch(S, batch)

            # Apply differentiable pooling
            x_pooled, adj_pooled, link_loss, ent_loss = dense_diff_pool(
                x_dense, adj_dense, S_dense, mask)

            # Flatten pooled features (1st coarsening level)
            graph_embeddings = x_pooled.mean(
                dim=1)  # [num_graphs, embedding_dim]

        elif self.pooling_method == "sagpool":
            x, edge_index, _, batch, _, _ = self.sagpool(
                x, edge_index, None, batch=batch)
            graph_embeddings = global_mean_pool(x, batch)

        else:
            raise ValueError(
                f"Unsupported pooling method: {self.pooling_method}")

        #  Prediction logits 
        logits = self.pred_head(graph_embeddings)  # [num_graphs, num_classes]

        # For binary classification, optionally squeeze last dim to shape [num_graphs]
        if self.num_classes == 1:
            logits = logits.view(-1)

        return logits, graph_embeddings, node_embeddings

# This one accepts edge_weights as an argument in forward()
# to be used with integrated gradients
class FlexibleGNN2(torch.nn.Module):
    def __init__(self, model_type, num_layers,
                 input_dim, hidden_dim, embedding_dim,
                 dropout, num_classes=1, pooling_method="add",
                 difpool_clusters_ratio=0.3, sagpool_ratio=0.6,
                 gat_heads=4):

        super(FlexibleGNN2, self).__init__()

        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_method = pooling_method
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.difpool_clusters_ratio = difpool_clusters_ratio
        self.sagpool_ratio = sagpool_ratio
        self.gat_heads = gat_heads

        # GNN layers
        self.convs = torch.nn.ModuleList()
        conv_layer = {
            'GraphConv': GraphConv,
            'SAGEConv': SAGEConv,
            'GCNConv': GCNConv,
            'GATConv': GATv2Conv
        }.get(model_type, None)
        if conv_layer is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type == 'GATConv':
            def intermediate_conv(in_d, out_d): return conv_layer(
                in_d, out_d // gat_heads, heads=gat_heads, dropout=dropout)
            def final_conv(in_d, out_d): return conv_layer(
                in_d, out_d, heads=1, concat=False, dropout=dropout)
        else:
            intermediate_conv = conv_layer
            final_conv = conv_layer

        self.convs.append(intermediate_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(intermediate_conv(hidden_dim, hidden_dim))
        self.convs.append(final_conv(hidden_dim, embedding_dim))

        # GraphNorm layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(GraphNorm(hidden_dim))

        # DiffPool layers
        if pooling_method == "diffpool":
            self.diffpool_assign_1 = GraphConv(embedding_dim, embedding_dim)
            num_clusters = int(difpool_clusters_ratio * embedding_dim)
            self.diffpool_assign_2 = GraphConv(embedding_dim, num_clusters)

        # SAGPool layers
        if pooling_method == "sagpool":
            self.sagpool = SAGPooling(embedding_dim, ratio=sagpool_ratio)

        # Prediction head
        self.pred_head = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, data, node_features=None, use_edge_weights=False, edge_weights=None, add_loops=True):
        x = node_features if node_features is not None else data.x_taxonomy_onehot
        edge_index, batch = data.edge_index, data.batch
        num_nodes = data.num_nodes

        # If edge_weights are passed (DualStream/IG), use those. 
        # Otherwise, look at data.weight if use_edge_weights is True.
        if edge_weights is not None:
            current_weights = edge_weights
            if add_loops:
                edge_index, current_weights = add_self_loops(
                    edge_index, current_weights, fill_value=1.0, num_nodes=num_nodes
                )
        elif use_edge_weights and hasattr(data, "weight") and data.weight is not None:
            current_weights = data.weight
            if add_loops:
                edge_index, current_weights = add_self_loops(
                    edge_index, current_weights, fill_value=1.0, num_nodes=num_nodes
                )
        else:
            if add_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            current_weights = None

        for i in range(self.num_layers):
            if current_weights is not None and self.model_type not in ["SAGEConv", "GATConv"]:
                # Use absolute value for weights as in your original FlexibleGNN
                x = self.convs[i](x, edge_index, edge_weight=abs(current_weights))
            else:
                x = self.convs[i](x, edge_index)

            if i != self.num_layers - 1:
                x = self.bns[i](x, batch=batch)
                x = torch.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Node embeddings
        node_embeddings = x.clone()

        # Graph embeddings
        if self.pooling_method == "add":
            graph_embeddings = global_add_pool(x, batch)
        elif self.pooling_method == "mean":
            graph_embeddings = global_mean_pool(x, batch)
        elif self.pooling_method == "max":
            graph_embeddings = global_max_pool(x, batch)
        elif self.pooling_method == "diffpool":
            x_dense, mask = to_dense_batch(x, batch)
            adj_dense = to_dense_adj(edge_index, batch=batch)
            S = F.relu(self.diffpool_assign_1(x, edge_index))
            S = self.diffpool_assign_2(S, edge_index)
            S_dense, _ = to_dense_batch(S, batch)
            x_pooled, _, _, _ = dense_diff_pool(x_dense, adj_dense, S_dense, mask)
            graph_embeddings = x_pooled.mean(dim=1)
        elif self.pooling_method == "sagpool":
            x, edge_index, _, batch, _, _ = self.sagpool(x, edge_index, None, batch=batch)
            graph_embeddings = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

        # Prediction logits
        logits = self.pred_head(graph_embeddings)
        if self.num_classes == 1:
            logits = logits.view(-1)

        return logits, graph_embeddings, node_embeddings

class DualStreamGNN(torch.nn.Module):
    def __init__(self, model_type, num_layers, input_dim, hidden_dim, embedding_dim, aggr_strategy = "concat", scale_weights = True, **kwargs):
        super(DualStreamGNN, self).__init__()

        self.aggr_strategy = aggr_strategy
        self.num_classes = kwargs.get('num_classes', 1)

        
        # Two instances of FlexibleGNN for positive and negative edge streams
        self.pos_stream = FlexibleGNN(model_type, num_layers, input_dim, 
                                      hidden_dim, embedding_dim, **kwargs)
        self.neg_stream = FlexibleGNN(model_type, num_layers, input_dim, 
                                      hidden_dim, embedding_dim, **kwargs)

        
        
        # Final prediction layer that takes concatenated embeddings from both streams
        if aggr_strategy == "concat":
            self.fusion_dim = embedding_dim * 2 
        else:
            self.fusion_dim = embedding_dim

        self.final_pred = torch.nn.Linear(self.fusion_dim, kwargs.get('num_classes', 1))

        # Initialize buffers for global min and max edge weights (for normalization)
        self.register_buffer('pos_min', torch.tensor(0.0))
        self.register_buffer('pos_max', torch.tensor(1.0))
        self.register_buffer('neg_min', torch.tensor(0.0))
        self.register_buffer('neg_max', torch.tensor(1.0))

        # Scale weights flag (can be turned off for ablation)
        self.scale_weights = scale_weights

    @torch.no_grad()
    def fit_global_scaling(self, train_loader):
        all_pos_weights = []
        all_neg_weights = []

        for data in train_loader:
            weights = data.weight
            all_pos_weights.append(weights[weights > 0])
            all_neg_weights.append(weights[weights < 0].abs())

        pos_all = torch.cat(all_pos_weights)
        neg_all = torch.cat(all_neg_weights)

        self.pos_min.fill_(pos_all.min())
        self.pos_max.fill_(pos_all.max())
        self.neg_min.fill_(neg_all.min())
        self.neg_max.fill_(neg_all.max())

        print(f"Fitted global scaling: pos_min={self.pos_min.item():.4f},pos_max={self.pos_max.item():.4f}, neg_min={self.neg_min.item():.4f}, neg_max={self.neg_max.item():.4f}")

    def forward(self, data, node_features=None, use_edge_weights=True):
            edge_index = data.edge_index
            edge_weight = data.weight
            num_nodes = data.num_nodes

            # get rid of previous self-loops for separate processing
            loop_mask = edge_index[0] == edge_index[1]
            edge_index_raw = edge_index[:, ~loop_mask]
            edge_weight_raw = edge_weight[~loop_mask]

            # Separate and Scale 'Real' Edges
            pos_m = edge_weight_raw > 0
            neg_m = edge_weight_raw < 0

            if use_edge_weights:
                if self.scale_weights:
                    # Scale Positive
                    pos_w = (edge_weight_raw[pos_m] - self.pos_min) / (self.pos_max - self.pos_min + 1e-8)
                    pos_w = torch.clamp(pos_w, 0.0, 1.0)
                    
                    # Scale Negative
                    neg_w = (edge_weight_raw[neg_m].abs() - self.neg_min) / (self.neg_max - self.neg_min + 1e-8)
                    neg_w = torch.clamp(neg_w, 0.0, 1.0)
                else:
                    # use raw weights (after separating pos/neg)
                    pos_w = edge_weight_raw[pos_m].float()  
                    neg_w = edge_weight_raw[neg_m].abs().float()  # Use absolute value for negative weights in the negative stream
            else:
                # Binary mask (1 for presence of edge, 0 for absence) if not using weights
                pos_w = torch.ones(pos_m.sum(), device=edge_weight.device)
                neg_w = torch.ones(neg_m.sum(), device=edge_weight.device)

            # add Self-Loops with specific values
            # Positive Stream: Loop weight = 1.0
            pos_idx, pos_w = add_self_loops(edge_index_raw[:, pos_m], pos_w, 
                                            fill_value=1.0, num_nodes=num_nodes)
            
            # Negative Stream: Loop weight = 0.0 (Node is not negatively connected to itself)
            neg_idx, neg_w = add_self_loops(edge_index_raw[:, neg_m], neg_w, 
                                            fill_value=0.0, num_nodes=num_nodes)

            # Prepare Stream Data
            pos_data = data.clone()
            pos_data.edge_index, pos_data.weight = pos_idx, pos_w
            
            neg_data = data.clone()
            neg_data.edge_index, neg_data.weight = neg_idx, neg_w

            # Forward through streams
            # add_loops=False because they are already added with specific weights above
            _, pos_graph_emb, _ = self.pos_stream(pos_data, node_features=node_features, use_edge_weights=use_edge_weights, add_loops=False)
            _, neg_graph_emb, _ = self.neg_stream(neg_data, node_features=node_features, use_edge_weights=use_edge_weights, add_loops=False)
            
            # 6. Fusion
            if self.aggr_strategy == "concat":
                combined_emb = torch.cat([pos_graph_emb, neg_graph_emb], dim=-1)
            elif self.aggr_strategy == "add":
                combined_emb = pos_graph_emb + neg_graph_emb
            elif self.aggr_strategy == "subtract":
                combined_emb = pos_graph_emb - neg_graph_emb
            
            logits = self.final_pred(combined_emb)
            if self.final_pred.out_features == 1:
                logits = logits.view(-1)

            # Logic for dummy embedding device
            device = node_features.device if node_features is not None else edge_weight.device
            dummy_node_emb = torch.zeros(num_nodes, self.pos_stream.embedding_dim, device=device)
                        
            return logits, combined_emb, dummy_node_emb


# This version of the DualStreamGNN class is designed to work with edge weights passed directly to the forward method. To be used with Integrated Gradients
class DualStreamGNN2(torch.nn.Module):
    def __init__(self, model_type, num_layers, input_dim, hidden_dim, embedding_dim, aggr_strategy="concat", **kwargs):
        super(DualStreamGNN2, self).__init__()

        self.aggr_strategy = aggr_strategy
        self.num_classes = kwargs.get('num_classes', 1)
        self.embedding_dim = embedding_dim

        # Initialize both streams using FlexibleGNN2
        self.pos_stream = FlexibleGNN2(model_type, num_layers, input_dim, 
                                       hidden_dim, embedding_dim, **kwargs)
        self.neg_stream = FlexibleGNN2(model_type, num_layers, input_dim, 
                                       hidden_dim, embedding_dim, **kwargs)

        if aggr_strategy == "concat":
            self.fusion_dim = embedding_dim * 2 
        else:
            self.fusion_dim = embedding_dim

        self.final_pred = torch.nn.Linear(self.fusion_dim, self.num_classes)

        # Buffers for global scaling
        self.register_buffer('pos_min', torch.tensor(0.0))
        self.register_buffer('pos_max', torch.tensor(1.0))
        self.register_buffer('neg_min', torch.tensor(0.0))
        self.register_buffer('neg_max', torch.tensor(1.0))

    @torch.no_grad()
    def fit_global_scaling(self, train_loader):
        all_pos_weights = []
        all_neg_weights = []
        for data in train_loader:
            weights = data.weight
            all_pos_weights.append(weights[weights > 0])
            all_neg_weights.append(weights[weights < 0].abs())
        
        pos_all = torch.cat(all_pos_weights)
        neg_all = torch.cat(all_neg_weights)
        self.pos_min.fill_(pos_all.min())
        self.pos_max.fill_(pos_all.max())
        self.neg_min.fill_(neg_all.min())
        self.neg_max.fill_(neg_all.max())

    def forward(self, data, node_features=None, use_edge_weights=True, edge_weights=None):
        # Critical: Use passed edge_weights (from IG) or fallback to data.weight
        weights = edge_weights if edge_weights is not None else data.weight
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Critical: Remove existing self-loops to prevent interference with stream-specific loop logic
        loop_mask = edge_index[0] == edge_index[1]
        edge_index_raw = edge_index[:, ~loop_mask]
        weights_raw = weights[~loop_mask]

        # Differentiable splitting of weights
        pos_w_raw = torch.relu(weights_raw)
        neg_w_raw = torch.relu(-weights_raw)

        if use_edge_weights:
            # Scale and clamp based on fitted range
            pos_w = torch.clamp((pos_w_raw - self.pos_min) / (self.pos_max - self.pos_min + 1e-8), 0.0, 1.0)
            neg_w = torch.clamp((neg_w_raw - self.neg_min) / (self.neg_max - self.neg_min + 1e-8), 0.0, 1.0)
        else:
            pos_w = (pos_w_raw > 0).float()
            neg_w = (neg_w_raw > 0).float()

        # Add self-loops with stream-specific fill values
        pos_idx, pos_w_final = add_self_loops(edge_index_raw, pos_w, fill_value=1.0, num_nodes=num_nodes)
        neg_idx, neg_w_final = add_self_loops(edge_index_raw, neg_w, fill_value=0.0, num_nodes=num_nodes)

        # Prepare stream data clones
        pos_data = data.clone()
        pos_data.edge_index = pos_idx
        
        neg_data = data.clone()
        neg_data.edge_index = neg_idx

        # Forward through streams with add_loops=False as they are handled here
        _, pos_graph_emb, _ = self.pos_stream(pos_data, node_features=node_features, 
                                              use_edge_weights=use_edge_weights, 
                                              edge_weights=pos_w_final, add_loops=False)
        
        _, neg_graph_emb, _ = self.neg_stream(neg_data, node_features=node_features, 
                                              use_edge_weights=use_edge_weights, 
                                              edge_weights=neg_w_final, add_loops=False)

        # Fusion logic
        if self.aggr_strategy == "concat":
            combined_emb = torch.cat([pos_graph_emb, neg_graph_emb], dim=-1)
        elif self.aggr_strategy == "subtract":
            combined_emb = pos_graph_emb - neg_graph_emb
        else:
            combined_emb = pos_graph_emb + neg_graph_emb

        logits = self.final_pred(combined_emb)
        if self.num_classes == 1:
            logits = logits.view(-1)

        # Consistency return
        dummy_node_emb = torch.zeros(num_nodes, self.embedding_dim, device=weights.device)

        return logits, combined_emb, dummy_node_emb