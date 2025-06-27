from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import Dataset, PgDatasetConfig
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
import numpy as np
import torch
from polymer_gc.model.base import PolyGCBaseModel
from torch import optim
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from itertools import cycle

def make_bidirectional(edge_index):
    """
    Convert unidirectional edges to bidirectional by adding reverse edges.
    
    Args:
        edge_index: torch.Tensor of shape [2, num_edges] with source and target nodes
        
    Returns:
        torch.Tensor: Bidirectional edge_index with shape [2, num_edges*2]
    """
    # Get the reverse edges (swap source and target)
    reverse_edges = edge_index.flip(0)
    
    # Concatenate original and reverse edges
    bidirectional_edges = torch.cat([edge_index, reverse_edges], dim=1)
    
    # Remove duplicates (in case the original graph already had some bidirectional edges)
    # Convert to tuples for easier deduplication
    edge_tuples = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
    reverse_tuples = [(int(reverse_edges[0, i]), int(reverse_edges[1, i])) for i in range(reverse_edges.shape[1])]
    
    # Combine and deduplicate
    all_edges = list(set(edge_tuples + reverse_tuples))
    
    # Convert back to tensor
    if all_edges:
        bidirectional_edges = torch.tensor(all_edges, dtype=torch.long).T
    else:
        bidirectional_edges = torch.empty((2, 0), dtype=torch.long)
    
    return bidirectional_edges

SEED=42
EPOCHS=0
DPI=600
np.random.seed(SEED)
torch.manual_seed(SEED)

db_path = "database.db"
main_dir = Path(__file__).parent / Path(__file__).stem
main_dir.mkdir(parents=True, exist_ok=True)
data_dir = main_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

with SessionManager(db_path) as session:
    # Assuming this setup code is correct and populates the DB
    # for smiles in tqdm(lin_monomers):
    #     SQLStructureModel.get_or_create(smiles=smiles, commit=False)
    # session.commit()

    dataset = Dataset.get(name="RandomArchitecture")
    target_names=dataset.config.targets

graph_data_file = data_dir / "graph_data.pt"
if graph_data_file.exists():
    all_graph_data = torch.load(graph_data_file, weights_only=False)
else:

    with SessionManager(db_path) as session:
        # Assuming this setup code is correct and populates the DB
        # for smiles in tqdm(lin_monomers):
        #     SQLStructureModel.get_or_create(smiles=smiles, commit=False)
        # session.commit()

        dataset = Dataset.get(name="RandomArchitecture")
        data = dataset.load_entries_data()


    strucid_to_idx = {val: idx for idx, val in enumerate(data["structure_ids"])}
    vec_strucid_to_idx = np.vectorize(strucid_to_idx.get)
    target_names = list(data["targets"].keys())
    targets_array = np.stack([
        data["targets"][n] 
        for n in target_names
    ]).T
    print(data["targets"])
    print(targets_array.shape)

    all_graph_data=[]
    for g in tqdm(data["graphs"], desc="Loading graphs"):
        
        structure_idx = vec_strucid_to_idx(g["nodes"])
        embeddings = data["all_embeddings"][structure_idx]
        
        # Make edges bidirectional
        edges = torch.tensor(g["edges"], dtype=torch.long).T
        bidirectional_edges = make_bidirectional(edges)

        graph_data = Data(
            x=torch.tensor(embeddings, dtype=torch.float32),
            edge_index=bidirectional_edges,  
            y=torch.tensor(np.atleast_2d(targets_array[g["entry_pos"]]), dtype=torch.float32),
            entry_pos=g["entry_pos"], # IMPORTANT: Keep track of the original entry
        )
        all_graph_data.append(graph_data)

    print(f"Total number of graphs loaded: {len(all_graph_data)}")

    with open(graph_data_file, "wb") as f:
        torch.save(all_graph_data, f)


entry_indices = np.unique([g.entry_pos for g in all_graph_data])
num_entries = len(entry_indices)
np.random.RandomState(SEED).shuffle(entry_indices)

train_split, val_split = 0.8, 0.9
train_idx_limit = int(num_entries * train_split)
val_idx_limit = int(num_entries * val_split)

train_entry_indices = set(entry_indices[:train_idx_limit])
val_entry_indices = set(entry_indices[train_idx_limit:val_idx_limit])
test_entry_indices = set(entry_indices[val_idx_limit:])

train_graphs = [g for g in all_graph_data if g.entry_pos in train_entry_indices]
val_graphs = [g for g in all_graph_data if g.entry_pos in val_entry_indices]
test_graphs = [g for g in all_graph_data if g.entry_pos in test_entry_indices]

print(f"Splitting based on {num_entries} entries:")
print(f"  Train graphs: {len(train_graphs)} (from {len(train_entry_indices)} entries)")
print(f"  Validation graphs: {len(val_graphs)} (from {len(val_entry_indices)} entries)")
print(f"  Test graphs: {len(test_graphs)} (from {len(test_entry_indices)} entries)")

# Create DataLoaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes_per_task = [len(dataset.config.target_classes[target]) for target in target_names]  

model_config = PolyGCBaseModel.ModelConfig(
    task_type="classification",
    monomer_features=64, # Must match dummy data
    num_classes_per_task=num_classes_per_task, # From setup
    num_gnn_layers=3,
    mlp_layer=2,
    dropout_rate=0.2,
    mass_distribution_buckets=0,
    pooling_layers=[{"type": "mean"}, {"type": "max"}]
)

model = PolyGCBaseModel(config=model_config)

model_file = main_dir / "model.pt"
if model_file.exists():
    try:
        model.load_state_dict(torch.load(model_file))
    except Exception as e:
        print(f"Error loading model: {e}")
        

model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)

print("\n--- Starting Training ---")
for epoch in range(EPOCHS): # Train for a few epochs
    try:
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.batch_loss(batch, 'train')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0
        correct_preds_task1, correct_preds_task2 = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating epoch {epoch+1}"):
                batch = batch.to(device)
                total_val_loss += model.batch_loss(batch, 'val').item() * batch.num_graphs
                preds = model.predict(batch)
                
                correct_preds_task1 += (preds[:, 0] == batch.y[:, 0].long()).sum().item()
                correct_preds_task2 += (preds[:, 1] == batch.y[:, 1].long()).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        acc1 = correct_preds_task1 / len(val_loader.dataset)
        acc2 = correct_preds_task2 / len(val_loader.dataset)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc (T1): {acc1:.2%} | Val Acc (T2): {acc2:.2%}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
        break

torch.save(model.state_dict(), model_file)





import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from itertools import cycle

# =============================================================================
# --- Step 3: Visualization of Model Performance on Test Set ---
# =============================================================================
print("\n--- Generating Visualizations for the Test Set ---")

# First, get the actual class names for labeling plots
# This assumes your targets are integer-encoded.
# If you have a mapping from index to name, use that. Otherwise, we'll generate them.
class_names_per_task = []
for i, target_name in enumerate(target_names):
    # Try to get from config, otherwise generate generic names
    if hasattr(dataset.config, 'target_classes') and target_name in dataset.config.target_classes:
        class_names_per_task.append(dataset.config.target_classes[target_name])
    else:
        # Generate generic class names like "Class 0", "Class 1", etc.
        num_classes = model.config.num_classes_per_task[i]
        class_names_per_task.append([f"Class {j}" for j in range(num_classes)])

# --- Collect all predictions, probabilities, and true labels from the test set ---
model.eval()
all_true_labels = []
all_preds = []
all_probas = [[] for _ in target_names]
all_embeddings = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        batch = batch.to(device)
        # Store true labels
        all_true_labels.append(batch.y.cpu())
        
        # Get discrete predictions
        preds = model.predict(batch)
        all_preds.append(preds.cpu())
        
        # Get class probabilities for ROC curves
        probas = model.predict_proba(batch)
        for i in range(len(target_names)):
            all_probas[i].append(probas[i].cpu())
        
        # Get graph embeddings for t-SNE plot
        embeddings = model.predict_embedding(batch)
        all_embeddings.append(embeddings.cpu())

# Concatenate all batch results into single tensors/arrays
y_true_all = torch.cat(all_true_labels, dim=0).numpy()
y_pred_all = torch.cat(all_preds, dim=0).numpy()
y_probas_all = [torch.cat(p, dim=0).numpy() for p in all_probas]
embeddings_all = torch.cat(all_embeddings, dim=0).numpy()

result_dir = main_dir / "results"
result_dir.mkdir(parents=True, exist_ok=True)
# --- Visualization Helper Functions ---

def plot_confusion_matrix(y_true, y_pred, class_names, task_name):
    """Plots a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for: {task_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(result_dir / f"confusion_matrix_{task_name}.png", dpi=DPI)
    plt.close()

def plot_roc_curves(y_true, y_probas, class_names, task_name):
    """Plots multiclass ROC curves using the One-vs-Rest strategy."""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Multi-class ROC Curves for: {task_name}', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_dir / f"roc_curve_{task_name}.png", dpi=DPI)
    plt.close()

def plot_tsne_embeddings(embeddings, labels, class_names, task_name):
    """Plots a 2D t-SNE projection of graph embeddings."""
    print(f"Running t-SNE for {task_name}... (this may take a moment)")
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=300, random_state=SEED)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=labels,
        palette=sns.color_palette("hsv", len(class_names)),
        legend="full",
        alpha=0.8
    )
    
    # Manually set legend labels to be the actual class names
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, class_names, title="Classes")
    
    plt.title(f't-SNE Projection of Graph Embeddings, Colored by True {task_name}', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(result_dir / f"tsne_embedding_{task_name}.png", dpi=DPI)
    plt.close()

# --- Generate and Display Plots for Each Task ---

for i, task_name in enumerate(target_names):
    print(f"\n{'='*20} ANALYSIS FOR TASK: {task_name} {'='*20}")
    
    y_true_task = y_true_all[:, i]
    y_pred_task = y_pred_all[:, i]
    y_probas_task = y_probas_all[i]
    class_names_task = class_names_per_task[i]
    
    # 1. Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true_task, y_pred_task, target_names=class_names_task)
    print(report)
    with open(result_dir / f"classification_report_{task_name}.txt", "w") as f:
        f.write(report)
    
    # 2. Confusion Matrix
    plot_confusion_matrix(y_true_task, y_pred_task, class_names_task, task_name)
    
    # 3. ROC Curves
    plot_roc_curves(y_true_task, y_probas_task, class_names_task, task_name)
    
    # 4. t-SNE Plot
    plot_tsne_embeddings(embeddings_all, y_true_task, class_names_task, task_name)



import pandas as pd





y_true_all = torch.cat(all_true_labels, dim=0).numpy()
y_pred_all = torch.cat(all_preds, dim=0).numpy()
y_probas_all = [torch.cat(p, dim=0).numpy() for p in all_probas]
embeddings_all = torch.cat(all_embeddings, dim=0).numpy()

result_dir = main_dir / "results2"
result_dir.mkdir(parents=True, exist_ok=True)

# --- NEW: Combined Visualization Function ---

def plot_combined_tsne_with_errors(
    embeddings, y_true, y_pred, class_names_per_task, target_names, result_dir, SEED=42
):
    """
    Generates a single, information-rich t-SNE plot that combines multiple attributes:
    - Color represents the first task (e.g., architecture).
    - Marker style represents the second task (e.g., structure).
    - Misclassified points are highlighted with a red edge.
    """
    print("Running t-SNE for combined plot... (this may take a moment)")
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=500, random_state=SEED)
    tsne_results = tsne.fit_transform(embeddings)

    # --- Prepare a Pandas DataFrame for easier plotting with Seaborn ---
    # Get string names for labels
    arch_labels = [class_names_per_task[0][int(i)] for i in y_true[:, 0]]
    struct_labels = [class_names_per_task[1][int(i)] for i in y_true[:, 1]]
    
    # Determine which points were misclassified
    arch_correct = y_true[:, 0] == y_pred[:, 0]
    struct_correct = y_true[:, 1] == y_pred[:, 1]
    is_misclassified = ~ (arch_correct & struct_correct)

    df_tsne = pd.DataFrame({
        'tSNE-1': tsne_results[:, 0],
        'tSNE-2': tsne_results[:, 1],
        target_names[0]: arch_labels,      # e.g., 'hot_encoded_architecture'
        target_names[1]: struct_labels,     # e.g., 'hot_encoded_structure'
        'is_misclassified': is_misclassified
    })

    # --- Create the Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 15))

    # Plot all points first
    sns.scatterplot(
        x='tSNE-1',
        y='tSNE-2',
        hue=target_names[0],
        style=target_names[1],
        data=df_tsne,
        ax=ax,
        s=120,  # size of markers
        alpha=0.8,
        legend='full'
    )
    
    # Highlight the misclassified points by plotting them again with a red edge
    # misclassified_df = df_tsne[df_tsne['is_misclassified']]
    # if not misclassified_df.empty:
    #     ax.scatter(
    #         misclassified_df['tSNE-1'],
    #         misclassified_df['tSNE-2'],
    #         s=150,
    #         facecolors='none',
    #         edgecolors='red',
    #         linewidths=2,
    #         label='_nolegend_' # This point is just for visual effect, no legend entry
    #     )
    
    # --- Final Plot Adjustments ---
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Add a custom handle for the misclassification marker
    # from matplotlib.lines import Line2D
    # error_handle = Line2D([0], [0], marker='o', color='w', label='Misclassified',
    #                       markerfacecolor='none', markeredgecolor='red', markersize=10, markeredgewidth=2)
    # handles.append(error_handle)
    
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)
    
    ax.set_title(
        f"t-SNE of Graph Embeddings\nColor: {target_names[0].replace('_', ' ').title()} | Marker: {target_names[1].replace('_', ' ').title()}",
        fontsize=24,
        pad=20
    )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend
    
    # Save the plot
    save_path = result_dir / "combined_tsne_visualization.png"
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Combined visualization saved to: {save_path}")


# --- Generate and Display Plots for Each Task ---

# 1. Generate standard reports and confusion matrices for each task
for i, task_name in enumerate(target_names):
    print(f"\n{'='*20} ANALYSIS FOR TASK: {task_name} {'='*20}")
    
    y_true_task = y_true_all[:, i]
    y_pred_task = y_pred_all[:, i]
    class_names_task = class_names_per_task[i]
    
    # Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true_task, y_pred_task, target_names=class_names_task)
    print(report)
    with open(result_dir / f"classification_report_{task_name}.txt", "w") as f:
        f.write(report)
    
    # Confusion Matrix
    plot_confusion_matrix(y_true_task, y_pred_task, class_names_task, task_name)


# 2. Generate the new combined t-SNE plot
plot_combined_tsne_with_errors(
    embeddings_all,
    y_true_all,
    y_pred_all,
    class_names_per_task,
    target_names,
    result_dir,
    SEED=SEED
)





y_true_all = torch.cat(all_true_labels, dim=0).numpy()
y_pred_all = torch.cat(all_preds, dim=0).numpy()
y_probas_all = [torch.cat(p, dim=0).numpy() for p in all_probas]
embeddings_all = torch.cat(all_embeddings, dim=0).numpy()

result_dir = main_dir / "results3"
result_dir.mkdir(parents=True, exist_ok=True)

# --- Visualization Helper Functions ---


def plot_confusion_matrix(y_true, y_pred, class_names, task_name, result_dir):
    """Plots a heatmap of the confusion matrix based on counts."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix (Counts) for: {task_name}", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(result_dir / f"confusion_matrix_{task_name}.png", dpi=DPI)
    plt.close()


# --- NEW: PROBABILITY-BASED VISUALIZATIONS ---


def plot_probability_confusion_matrix(
    y_true, y_probas, class_names, task_name, result_dir
):
    """
    Plots a confusion matrix where each cell shows the average probability
    assigned to the predicted class.
    """
    num_classes = len(class_names)
    prob_cm = np.zeros((num_classes, num_classes))

    # Group probabilities by true class and calculate the mean for each predicted class
    for true_class_idx in range(num_classes):
        # Find all samples that belong to this true class
        mask = y_true == true_class_idx
        if np.sum(mask) > 0:
            # Get the probabilities for these samples and average them column-wise
            prob_cm[true_class_idx, :] = np.mean(y_probas[mask], axis=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        prob_cm,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Average Prediction Probability Matrix for: {task_name}", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(result_dir / f"probability_confusion_matrix_{task_name}.png", dpi=DPI)
    plt.close()


def plot_average_probability_bars(y_true, y_probas, class_names, task_name, result_dir):
    """
    For each true class, plots a stacked bar showing the average distribution
    of predicted probabilities across all possible classes.
    """
    num_classes = len(class_names)
    avg_probs_by_true_class = np.zeros((num_classes, num_classes))

    for true_class_idx in range(num_classes):
        mask = y_true == true_class_idx
        if np.sum(mask) > 0:
            avg_probs_by_true_class[true_class_idx] = np.mean(y_probas[mask], axis=0)

    df_probs = pd.DataFrame(
        avg_probs_by_true_class, index=class_names, columns=class_names
    )

    # Plotting the stacked bar chart
    ax = df_probs.plot(
        kind="bar", stacked=True, figsize=(12, 8), colormap="viridis", edgecolor="black"
    )

    plt.title(
        f"Average Predicted Probability Distribution per True Class: {task_name}",
        fontsize=16,
    )
    plt.xlabel("True Class", fontsize=12)
    plt.ylabel("Average Probability", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Predicted Class", bbox_to_anchor=(1.02, 1), loc="upper left") 
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(result_dir / f"avg_probability_bars_{task_name}.png", dpi=DPI)
    plt.close()


# --- Generate and Display Plots for Each Task ---

for i, task_name in enumerate(target_names):
    print(f"\n{'=' * 20} ANALYSIS FOR TASK: {task_name} {'=' * 20}")

    y_true_task = y_true_all[:, i]
    y_pred_task = y_pred_all[:, i]
    y_probas_task = y_probas_all[i]
    class_names_task = class_names_per_task[i]

    # 1. Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(
        y_true_task, y_pred_task, target_names=class_names_task
    )
    print(report)
    with open(result_dir / f"classification_report_{task_name}.txt", "w") as f:
        f.write(report)

    # 2. Standard Confusion Matrix (Counts)
    plot_confusion_matrix(
        y_true_task, y_pred_task, class_names_task, task_name, result_dir
    )

    # 3. NEW: Probability-based Confusion Matrix
    print(f"--- Generating probability confusion matrix for {task_name} ---")
    plot_probability_confusion_matrix(
        y_true_task, y_probas_task, class_names_task, task_name, result_dir
    )

    # 4. NEW: Average Probability Distribution Bars
    print(f"--- Generating average probability bars for {task_name} ---")
    plot_average_probability_bars(
        y_true_task, y_probas_task, class_names_task, task_name, result_dir
    )




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# [Keep all your code up to this point]
# ...
# Concatenate all batch results into single tensors/arrays
y_true_all = torch.cat(all_true_labels, dim=0).numpy()
y_pred_all = torch.cat(all_preds, dim=0).numpy()
y_probas_all = [torch.cat(p, dim=0).numpy() for p in all_probas]
embeddings_all = torch.cat(all_embeddings, dim=0).numpy()

result_dir = main_dir / "results4"
result_dir.mkdir(parents=True, exist_ok=True)


# --- NEW COMBINED VISUALIZATION FUNCTIONS ---

def plot_combined_prob_matrices(y_true_all, y_probas_all, class_names_per_task, target_names, result_dir):
    """
    Generates a single figure with side-by-side subplots of the same size,
    each showing the probability confusion matrix for one task, with a shared colorbar.
    """
    num_tasks = len(target_names)
    
    # figsize can be tweaked for better appearance
    fig = plt.figure(figsize=(8 * num_tasks, 7))
    
    # GridSpec provides control over the layout. The last column is for the colorbar.
    # `width_ratios` makes the colorbar column narrow. `wspace` adds space between heatmaps.
    gs = GridSpec(1, num_tasks + 1, width_ratios=[20] * num_tasks + [1], wspace=0.3)

    # Create subplots, sharing the Y-axis for a consistent look
    axes = []
    share_y_ax = None
    for i in range(num_tasks):
        ax = fig.add_subplot(gs[0, i], sharey=share_y_ax)
        if i == 0:
            share_y_ax = ax
        axes.append(ax)
    
    cbar_ax = fig.add_subplot(gs[0, num_tasks])
    
    fig.suptitle('Average Prediction Probability Matrices', fontsize=20)

    # Calculate global min and max probability to ensure consistent color scaling across all heatmaps
    all_prob_cms = []
    for i in range(num_tasks):
        y_true_task = y_true_all[:, i]
        y_probas_task = y_probas_all[i]
        class_names = class_names_per_task[i]
        num_classes = len(class_names)
        
        prob_cm = np.zeros((num_classes, num_classes))
        for true_class_idx in range(num_classes):
            mask = y_true_task == true_class_idx
            if np.sum(mask) > 0:
                prob_cm[true_class_idx, :] = np.mean(y_probas_task[mask], axis=0)
        all_prob_cms.append(prob_cm)

    vmin = min(cm.min() for cm in all_prob_cms)
    vmax = max(cm.max() for cm in all_prob_cms)

    for i, task_name in enumerate(target_names):
        ax = axes[i]
        prob_cm = all_prob_cms[i]
        class_names = class_names_per_task[i]
        
        # The last heatmap gets the colorbar
        is_last = (i == num_tasks - 1)
        sns.heatmap(prob_cm, annot=True, fmt='.2f', cmap='viridis', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    cbar=is_last,
                    cbar_ax=cbar_ax if is_last else None,
                    square=True, # This ensures each cell is square
                    vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Task: {task_name.replace('_', ' ').title()}", fontsize=16)

    # Use figure-centric labels for a clean look
    fig.supylabel('True Label', fontsize=12)
    fig.supxlabel('Predicted Label', fontsize=12)

    # `tight_layout` with `rect` prevents title/labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(result_dir / "combined_probability_confusion_matrix.png", dpi=DPI)
    plt.close()
    print(f"Saved combined probability matrix plot to {result_dir}")



def plot_combined_avg_prob_bars(y_true_all, y_probas_all, class_names_per_task, target_names, result_dir):
    """
    Generates a single grouped bar plot showing the average predicted probabilities
    for each true class, across two tasks.
    """
    all_rows = []
    
    for task_idx, task_name in enumerate(target_names):
        y_true_task = y_true_all[:, task_idx]
        y_probas_task = y_probas_all[task_idx]
        class_names = class_names_per_task[task_idx]
        num_classes = len(class_names)

        avg_probs = np.zeros((num_classes, num_classes))
        for true_class_idx in range(num_classes):
            mask = y_true_task == true_class_idx
            if np.sum(mask) > 0:
                avg_probs[true_class_idx] = np.mean(y_probas_task[mask], axis=0)

        for i, true_class in enumerate(class_names):
            for j, pred_class in enumerate(class_names):
                all_rows.append({
                    "True Class": true_class,
                    "Predicted Class": pred_class,
                    "Average Probability": avg_probs[i, j],
                    "Task": task_name.replace('_', ' ').title()
                })

    df = pd.DataFrame(all_rows)

    # Plot grouped bar chart using seaborn for clarity
    import seaborn as sns
    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df,
        x="True Class",
        y="Average Probability",
        hue="Predicted Class",
        # ci=None,
        # palette="viridis",
        # edgecolor="black"
    )

    plt.title("Average Predicted Probability Distribution per True Class (Combined Tasks)", fontsize=18)
    plt.xlabel("True Class", fontsize=12)
    plt.ylabel("Average Probability", fontsize=12)
    # plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Predicted Class",
                # bbox_to_anchor=(1.05, 1), loc='upper left'
                )
    plt.tight_layout()

    output_path = result_dir / "combined_avg_probability_bars.png"
    plt.savefig(output_path, dpi=DPI)   
    plt.close()
    print(f"Saved combined probability bar plot to {output_path}")


def plot_worst_predictions(
    test_graphs, 
    y_true_all, 
    y_pred_all, 
    y_probas_all, 
    class_names_per_task, 
    target_names,
    result_dir, 
    n_top=10
):
    """
    Identifies and plots the top N graphs that were most confidently misclassified,
    coloring the nodes based on their feature vectors to show monomer distribution.
    """
    print(f"\n--- Identifying and plotting the {n_top} most confident misclassifications ---")
    
    errors = []
    # We must iterate through the original test set to link back to the graph data
    for i in range(len(test_graphs)):
        is_misclassified = False
        error_prob = -1.0 
        error_details = {}

        # Check Task 1 (e.g., Architecture)
        true_label_t1 = int(y_true_all[i, 0])
        pred_label_t1 = int(y_pred_all[i, 0])
        if true_label_t1 != pred_label_t1:
            is_misclassified = True
            prob = y_probas_all[0][i, pred_label_t1]
            if prob > error_prob:
                error_prob = prob
                error_details = {
                    'task_idx': 0,
                    'true_label': true_label_t1,
                    'pred_label': pred_label_t1,
                    'true_prob': y_probas_all[0][i, true_label_t1]
                }

        # Check Task 2 (e.g., Structure)
        true_label_t2 = int(y_true_all[i, 1])
        pred_label_t2 = int(y_pred_all[i, 1])
        if true_label_t2 != pred_label_t2:
            is_misclassified = True
            prob = y_probas_all[1][i, pred_label_t2]
            if prob > error_prob:
                error_prob = prob
                error_details = {
                    'task_idx': 1,
                    'true_label': true_label_t2,
                    'pred_label': pred_label_t2,
                    'true_prob': y_probas_all[1][i, true_label_t2]
                }
        
        if is_misclassified:
            errors.append({
                'test_set_index': i,
                'error_prob': error_prob,
                'details': error_details
            })

    # Sort all errors by the confidence in the wrong prediction (descending)
    top_errors = sorted(errors, key=lambda x: x['error_prob'], reverse=True)[:n_top]

    # --- Plotting ---
    if not top_errors:
        print("No misclassifications found to plot.")
        return
        
    fig, axes = plt.subplots(5, 2, figsize=(20, 45))
    axes = axes.flatten()
    fig.suptitle(f'Top {len(top_errors)} Most Confident Misclassifications', fontsize=24)

    for i, error_info in enumerate(top_errors):
        if i >= len(axes): break # Ensure we don't go out of bounds
        ax = axes[i]
        
        # Retrieve original graph data
        graph_data = test_graphs[error_info['test_set_index']]
        G = to_networkx(graph_data, to_undirected=True)
        
        # Prepare title with rich information
        details = error_info['details']
        task_idx = details['task_idx']
        task_name = target_names[task_idx].replace('_', ' ').title()
        
        true_name = class_names_per_task[task_idx][details['true_label']]
        pred_name = class_names_per_task[task_idx][details['pred_label']]
        
        title = (
            f"#{i+1}: Confidently Wrong on '{task_name}'\n"
            f"Predicted: '{pred_name}' (Prob: {error_info['error_prob']:.1%})\n"
            f"True: '{true_name}' (Prob: {details['true_prob']:.1%})"
        )
        
        # --- MODIFIED NODE COLORING ---
        # Get node features from the graph data object
        node_features = graph_data.x
        # Reduce each feature vector to a single scalar value (e.g., the mean).
        # This scalar acts as a proxy for the monomer type.
        scalar_node_values = torch.mean(node_features, dim=1).numpy()
        
        # Use a colormap to visualize the distribution of monomer types.
        # NetworkX handles the normalization and mapping automatically.
        cmap = plt.get_cmap('viridis')
        
        pos = nx.kamada_kawai_layout(G)
        nx.draw(
            G, 
            pos, 
            ax=ax, 
            with_labels=False, 
            node_size=60, 
            width=1.5,
            node_color=scalar_node_values, # Pass scalar values for coloring
            cmap=cmap                     # Specify the colormap
        )
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis('off')
        
    # Hide any unused subplots
    for j in range(len(top_errors), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(result_dir / "worst_10_predictions_colored_nodes.png", dpi=300)
    plt.close()
    print(f"Saved worst predictions plot with colored nodes to {result_dir}")


# --- Main Analysis Script ---

# 1. Generate standard reports and confusion matrices for each task (optional, but good for logs)
for i, task_name in enumerate(target_names):
    print(f"\n{'='*20} ANALYSIS FOR TASK: {task_name} {'='*20}")
    
    y_true_task = y_true_all[:, i]
    y_pred_task = y_pred_all[:, i]
    class_names_task = class_names_per_task[i]
    
    report = classification_report(y_true_task, y_pred_task, target_names=class_names_task)
    print(report)
    with open(result_dir / f"classification_report_{task_name}.txt", "w") as f:
        f.write(report)

# 2. Generate the new combined plots
print("\n--- Generating Combined Probability Visualizations ---")
plot_combined_prob_matrices(y_true_all, y_probas_all, class_names_per_task, target_names, result_dir)
plot_combined_avg_prob_bars(y_true_all, y_probas_all, class_names_per_task, target_names, result_dir)
plot_worst_predictions(
    test_graphs, 
    y_true_all, 
    y_pred_all, 
    y_probas_all, 
    class_names_per_task, 
    target_names,
    result_dir
)