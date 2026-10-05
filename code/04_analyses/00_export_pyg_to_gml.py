import torch
import networkx as nx
import sys
from torch_geometric.utils import to_networkx
from pathlib import Path

# --- Setup Paths ---
# Locate the project root (works from the terminal and interactively) and load shared paths
_start = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_root = next(p for p in (_start, *_start.parents) if (p / "code" / "common" / "paths.py").exists())
sys.path.insert(0, str(_root / "code" / "common"))
import paths
import helper_functions  # noqa: F401  (defines GraphDictDataset, needed by torch.load to read the saved datasets)

PYG_DATASET_ROOT = paths.DATA_PYG

# Define the list of subfolders within PYG_DATASET_ROOT to process.
DATASET_SUBFOLDERS = [
    "HMP_timepoints_self_w_functions_onehot",
    "Hall_timepoints_self_w_functions_onehot"
]

def export_pyg_to_gml(pyg_dataset_path: Path):
    print(f"\n--- Processing Dataset: {pyg_dataset_path.name} ---")

    # Define Output Directory: .../parent_folder/gml/assoc_t_str/
    base_dir = pyg_dataset_path.parent
    parts = pyg_dataset_path.stem.split('_')
    
    # Extract the threshold name (e.g., 'assoc_0_4'). Assumes it's the last two parts.
    threshold_dir_name = f"{parts[-2]}_{parts[-1]}"
    
    output_dir = base_dir / "gml" / threshold_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting GML files to: {output_dir}")

    # Load the PyG Dataset
    try:
        # Load with weights_only=False
        dataset = torch.load(pyg_dataset_path, weights_only=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Iterate and Export Each Graph
    total_graphs = len(dataset)
    for i, data in enumerate(dataset):
        try:
            # Convert PyG Data to NetworkX Graph
            # Exclude large feature tensors ('x_...')
            node_attributes = [
                attr for attr in dir(data) 
                if not attr.startswith('_') and attr not in ['edge_index', 'pos', 'face'] 
                and not attr.startswith('x_') and isinstance(getattr(data, attr), torch.Tensor) 
            ]

            G = to_networkx(
                data, 
                node_attrs=node_attributes, 
                edge_attrs=['weight'] if hasattr(data, 'weight') else None,
                to_undirected=True
            )

            if hasattr(data, 'node_name'):
                names = getattr(data, 'node_name') 
                
                for node_id, node_name in enumerate(names):
                    # Save to 'name': This is igraph's primary identifier (V(g)$name)
                    G.nodes[node_id]['name'] = str(node_name) 
                    
                    # Save to 'label': Keeps compatibility with standard GML viewers/readers
                    G.nodes[node_id]['label'] = str(node_name)
            
            # Add Custom Graph Attributes
            true_label = data.y_binary.item() if hasattr(data, 'y_binary') else 0
            graph_name = getattr(data, 'name', f"graph_{i+1:04d}")
            
            G.graph['true_label'] = true_label
            G.graph['graph_name'] = graph_name
            G.graph['original_dataset'] = pyg_dataset_path.name

            # Write to GML file
            gml_path = output_dir / f"{graph_name}.gml"
            nx.write_gml(G, gml_path.as_posix())
            
            if (i + 1) % 100 == 0 or (i + 1) == total_graphs:
                print(f"  Exported {i+1}/{total_graphs} graphs...")

        except Exception as e:
            print(f"Error exporting graph {i+1}: {e}. Skipping.")
            continue


# --- Main Execution ---
print("Starting PyG to GML export...")
print(f"PyG Data Root: {PYG_DATASET_ROOT}")

found_datasets = []
for subfolder in DATASET_SUBFOLDERS:
    folder_path = PYG_DATASET_ROOT / subfolder
    # Find all .pt files inside the current subfolder
    found_datasets.extend(list(folder_path.glob("*.pt")))

if not found_datasets:
    print("No PyG datasets (.pt files) found in the specified subfolders.")
else:
    for dataset_path in found_datasets:
        export_pyg_to_gml(dataset_path)

print("\n✅ All specified PyG datasets processed and converted to GML.")