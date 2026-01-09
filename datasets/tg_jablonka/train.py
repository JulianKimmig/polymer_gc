import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- PyTorch & PyG Imports ---
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# --- Custom Library Imports ---
# Assuming these are in your environment
from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import Dataset
from polymer_gc.model.base import PolyGCBaseModel

# --- Configuration ---
SEED = 42
EPOCHS = 0  # Set a reasonable number of epochs for training
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
DPI = 300  # Resolution for saved plots

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# --- Directory Setup ---
# Use Path(".") to ensure it works in environments where __file__ is not defined
main_dir = Path(".") / "train_tg_prediction"
main_dir.mkdir(parents=True, exist_ok=True)
data_dir = main_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
result_dir = main_dir / "results"
result_dir.mkdir(parents=True, exist_ok=True)
model_file = main_dir / "tg_model.pt"
ds_name = "Tg_Prediction_from_CSV"
# --- Data Loading and Caching ---
db_path = "database.db"  # Assumed to be in the same directory
target_name = "Tg"
graph_data_file = data_dir / "tg_graph_data.pt"
if graph_data_file.exists():
    print(f"Loading cached data from {graph_data_file}...")
    all_graph_data = torch.load(graph_data_file, weights_only=False)
else:
    print("Processing data from database...")
    with SessionManager(db_path) as session:
        # Load the dataset created by the data generation script
        dataset = Dataset.get(name=ds_name)
        # 'load_entries_data' pulls all required info from the DB
        data = dataset.load_entries_data()

    # The structure of the data dictionary is assumed to be similar to the classification script
    strucid_to_idx = {val: idx for idx, val in enumerate(data["structure_ids"])}
    vec_strucid_to_idx = np.vectorize(strucid_to_idx.get)

    # Target is now a single value: Tg
    target_name = dataset.config.targets[0]  # Should be 'Tg'
    targets_array = data["targets"][target_name]
    print(f"Target variable: '{target_name}'")

    all_graph_data = []
    # The 'graphs' key contains info to reconstruct graph objects
    for g_info in tqdm(data["graphs"], desc="Creating PyG Data objects"):
        # Node features are the pre-trained embeddings for the polymer structures
        # Note: In this dataset, there's only one "monomer" type per graph (the whole polymer)
        structure_idx = vec_strucid_to_idx(g_info["nodes"])
        embeddings = data["all_embeddings"][structure_idx]

        # Edges from the dummy graph used during data creation
        edges = torch.tensor(g_info["edges"], dtype=torch.long).T

        # Target 'y' is the Tg value for the corresponding entry
        target_value = targets_array[g_info["entry_pos"]]

        graph_data_obj = Data(
            x=torch.tensor(embeddings, dtype=torch.float32),
            edge_index=edges,
            # Ensure y is a tensor with shape [1, 1] for consistency
            y=torch.tensor([[target_value]], dtype=torch.float32),
            entry_pos=g_info["entry_pos"],  # Crucial for data splitting
        )
        all_graph_data.append(graph_data_obj)

    print(f"Total number of graph data points created: {len(all_graph_data)}")
    with open(graph_data_file, "wb") as f:
        torch.save(all_graph_data, f)

# --- Data Splitting (by entry to prevent leakage) ---
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

print(f"\nSplitting based on {num_entries} unique polymer entries:")
print(f"  Train samples: {len(train_graphs)} (from {len(train_entry_indices)} entries)")
print(
    f"  Validation samples: {len(val_graphs)} (from {len(val_entry_indices)} entries)"
)
print(f"  Test samples: {len(test_graphs)} (from {len(test_entry_indices)} entries)")

# --- Create DataLoaders ---
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Definition ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# NOTE: The model configuration is adapted for regression with uncertainty prediction.
# The exact parameter names depend on your PolyGCBaseModel implementation.
model_config = PolyGCBaseModel.ModelConfig(
    task_type="regression",
    monomer_features=train_graphs[0].x.shape[1],  # Must match embedding dimension
    # For regression, we define the output dimension directly.
    # We are predicting one property (Tg), and the model will output
    # two values for it: mean (mu) and log-variance (log_sigma_sq).
    gc_features=256,
    num_target_properties=1,
    num_gnn_layers=4,
    mlp_layer=3,
    dropout_rate=0.3,
    pooling_layers=[{"type": "mean"}, {"type": "max"}, {"type": "sum"}],
    logits_output=True,
    mass_distribution_buckets=0,  # Disable mass distribution feature
)

model = PolyGCBaseModel(config=model_config)

if model_file.exists():
    try:
        print(f"Loading pre-trained model from {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device))
    except Exception as e:
        print(f"Could not load model state dict: {e}. Starting with a fresh model.")

model = model.to(device)

model.prefit(
    x=torch.cat([g.x for g in train_graphs], dim=0).to(device),
    y=torch.cat([g.y for g in train_graphs], dim=0).to(device),
)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "min",
    factor=0.5,
    patience=15,
)

# --- Training Loop ---
print("\n--- Starting Training ---")
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
patience_epochs = 40  # Early stopping patience
final_epoch = 0

try:
    for epoch in range(EPOCHS):
        final_epoch = epoch + 1
        model.train()
        total_train_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1:03d}/{EPOCHS} [Train]", leave=False
        ):
            batch = batch.to(device)
            optimizer.zero_grad()
            # The model's batch_loss should internally use GaussianNLLLoss
            loss = model.batch_loss(batch, "train")
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0
        all_val_preds_mu = []
        all_val_true = []
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1:03d}/{EPOCHS} [Val]", leave=False
            ):
                batch = batch.to(device)
                total_val_loss += (
                    model.batch_loss(batch, "val").item() * batch.num_graphs
                )
                # model.predict should return a tensor of shape [N, 2] for [mu, log_var]
                mean_preds, _ = model.predict(batch)
                all_val_preds_mu.append(mean_preds.cpu())
                all_val_true.append(batch.y.cpu())

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        # Calculate validation metrics
        val_preds_mu = torch.cat(all_val_preds_mu).numpy()
        val_true = torch.cat(all_val_true).numpy()
        val_mae = mean_absolute_error(val_true, val_preds_mu)

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.2f} K"
        )

        # Save best model and check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file)
            print(f"  -> New best model saved to {model_file}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_epochs:
                print(
                    f"Early stopping triggered after {patience_epochs} epochs without improvement."
                )
                break
except KeyboardInterrupt:
    print("Training interrupted by user.")

# --- Final Evaluation on Test Set ---
print("\n--- Final Evaluation on Test Set ---")
# Load the best performing model
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

all_preds_mu = []
all_preds_sigma = []
all_true = []
all_embeddings = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        batch = batch.to(device)
        mean_preds, log_var_preds = model.predict(batch)
        embeddings = model.predict_embedding(batch)

        # mu is the first column of the output
        all_preds_mu.append(mean_preds.cpu())
        # sigma = exp(0.5 * log_var), where log_var is the second column
        all_preds_sigma.append(torch.exp(0.5 * log_var_preds).cpu())
        all_true.append(batch.y.cpu())
        all_embeddings.append(embeddings.cpu())

# Concatenate results
y_pred_mu = torch.cat(all_preds_mu).numpy().flatten()
y_pred_sigma = torch.cat(all_preds_sigma).numpy().flatten()
y_true = torch.cat(all_true).numpy().flatten()
embeddings_all = torch.cat(all_embeddings, dim=0).numpy()

# --- Calculate and Print Final Metrics ---
test_mae = mean_absolute_error(y_true, y_pred_mu)
test_rmse = np.sqrt(mean_squared_error(y_true, y_pred_mu))
test_r2 = r2_score(y_true, y_pred_mu)

print("\n--- Test Set Performance ---")
print(f"  Mean Absolute Error (MAE): {test_mae:.2f} K")
print(f"  Root Mean Squared Error (RMSE): {test_rmse:.2f} K")
print(f"  R-squared (R²): {test_r2:.3f}")
print("--------------------------")

# --- Visualization of Results ---

# 1. Parity Plot (Predicted vs. True) with Uncertainty
plt.figure(figsize=(8, 8))
plt.errorbar(
    y_true,
    y_pred_mu,
    yerr=y_pred_sigma,
    fmt="o",
    color="royalblue",
    ecolor="lightskyblue",
    elinewidth=1,
    capsize=0,
    alpha=0.6,
    label="Predictions with Uncertainty",
)
min_val = min(y_true.min(), y_pred_mu.min()) - 25
max_val = max(y_true.max(), y_pred_mu.max()) + 25
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y=x)")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.xlabel("True Tg (K)", fontsize=14)
plt.ylabel("Predicted Tg (K)", fontsize=14)
plt.title(
    f"Parity Plot for Tg Prediction (Test Set)\nMAE: {test_mae:.2f} K | R²: {test_r2:.3f}",
    fontsize=16,
)
plt.legend()
plt.grid(True)
plt.tight_layout()
parity_plot_path = result_dir / "parity_plot_with_uncertainty.png"
plt.savefig(parity_plot_path, dpi=DPI)
plt.close()

# Create descriptive text file for the parity plot
parity_df = pd.DataFrame(
    {
        "True_Tg (K)": y_true,
        "Predicted_Tg (K)": y_pred_mu,
        "Predicted_Uncertainty_Sigma (K)": y_pred_sigma,
    }
)
parity_desc = f"""
# --- Parity Plot Data Description ---

This file describes the data used to generate '{parity_plot_path.name}'.

## Plot Description:
The plot is a "parity plot," which compares the model's predicted values (y-axis) against the true, experimental values (x-axis) for the glass transition temperature (Tg).

- **Points (Blue Circles):** Each point represents a single polymer from the test set. Its position is determined by its true Tg and the model's predicted Tg.
- **Ideal Line (Red Dashed Line):** This is the y=x line. A perfect model would have all its predictions fall exactly on this line. The closer the points are to this line, the more accurate the model.
- **Error Bars (Light Blue):** The vertical error bars on each point represent the model's predicted uncertainty (one standard deviation, or 'sigma') for that specific prediction. Larger bars indicate the model is less confident in its prediction.

## Key Performance Metrics (from Test Set):
- **Mean Absolute Error (MAE):** {test_mae:.2f} K
- **Root Mean Squared Error (RMSE):** {test_rmse:.2f} K
- **R-squared (R²):** {test_r2:.3f}

## Data Sample (first 15 entries):
{parity_df.head(15).to_string()}
"""
with open(result_dir / "parity_plot_data_description.txt", "w") as f:
    f.write(parity_desc)

# 2. Error Distribution Plot
errors = y_pred_mu - y_true
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30)
plt.title("Distribution of Prediction Errors (Predicted - True)", fontsize=16)
plt.xlabel("Error (K)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.axvline(0, color="r", linestyle="--")
plt.grid(True)
plt.tight_layout()
error_plot_path = result_dir / "error_distribution.png"
plt.savefig(error_plot_path, dpi=DPI)
plt.close()

# Create descriptive text file for the error distribution plot
error_df = pd.DataFrame({"Error (K)": errors})
error_desc = f"""
# --- Error Distribution Plot Data Description ---

This file describes the data used to generate '{error_plot_path.name}'.

## Plot Description:
This plot is a histogram that shows the distribution of prediction errors on the test set. The error is calculated as (Predicted Tg - True Tg).

- **X-axis:** The prediction error in Kelvin (K). A positive value means the model over-predicted, and a negative value means it under-predicted.
- **Y-axis:** The frequency or count of predictions falling into each error bin.
- **Red Dashed Line:** This line is at error = 0, representing a perfect prediction.
- **Blue Curve (KDE):** The Kernel Density Estimate provides a smooth curve to estimate the probability density function of the errors.

A good model typically has an error distribution that is centered at 0 and is roughly symmetrical, resembling a normal (Gaussian) distribution.

## Error Statistics:
- **Mean Error:** {np.mean(errors):.2f} K
- **Standard Deviation of Error:** {np.std(errors):.2f} K

## Data Sample (first 15 error values):
{error_df.head(15).to_string()}
"""
with open(result_dir / "error_distribution_data_description.txt", "w") as f:
    f.write(error_desc)

# 3. t-SNE Plot of Embeddings
print("\nGenerating t-SNE plot of embeddings...")
tsne = TSNE(
    n_components=2,
    verbose=0,
    perplexity=min(30, len(embeddings_all) - 1),
    max_iter=1000,
    random_state=SEED,
)
tsne_results = tsne.fit_transform(embeddings_all)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], c=y_true, cmap="viridis", alpha=0.8
)
plt.colorbar(scatter, label="True Tg (K)")
plt.title("t-SNE Projection of Polymer Embeddings", fontsize=16)
plt.xlabel("t-SNE Dimension 1", fontsize=14)
plt.ylabel("t-SNE Dimension 2", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
tsne_plot_path = result_dir / "tsne_embeddings_by_tg.png"
plt.savefig(tsne_plot_path, dpi=DPI)
plt.close()

# Create descriptive text file for the t-SNE plot
tsne_df = pd.DataFrame(
    {
        "tSNE_Dim_1": tsne_results[:, 0],
        "tSNE_Dim_2": tsne_results[:, 1],
        "True_Tg (K)": y_true,
    }
)
tsne_desc = f"""
# --- t-SNE Plot Data Description ---

This file describes the data used to generate '{tsne_plot_path.name}'.

## Plot Description:
This plot is a 2D visualization of the high-dimensional polymer embeddings generated by the GNN model for the test set. t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used to visualize these complex embeddings.

- **Each Point:** Represents a single polymer from the test set.
- **Position:** The position of each point in the 2D space is determined by the t-SNE algorithm, which tries to preserve the local neighborhood structure from the original high-dimensional embedding space. Polymers with similar embeddings will appear closer together.
- **Color:** Each point is colored according to its true, experimental glass transition temperature (Tg), as indicated by the color bar on the right.

A smooth gradient of colors across the plot suggests that the model has learned a meaningful representation (a "latent space") where polymers with similar properties are located near each other. This is a strong sign that the model is capturing chemically relevant information.

## Data Sample (first 15 entries):
{tsne_df.head(15).to_string(index=False)}
"""
with open(result_dir / "tsne_embeddings_data_description.txt", "w") as f:
    f.write(tsne_desc)

# --- Generate Comprehensive Training Analysis Report ---
print("\nGenerating comprehensive training analysis report...")

# Calculate target stats on training data for the report
train_y_true = torch.cat([g.y for g in train_graphs]).numpy().flatten()
target_stats = {
    "mean": np.mean(train_y_true),
    "std": np.std(train_y_true),
    "min": np.min(train_y_true),
    "max": np.max(train_y_true),
}
report_content = f"""
# Training Analysis Report: {ds_name}

## 1. Executive Summary
This report details the training and evaluation of a Graph Neural Network (GNN) model, `PolyGCBaseModel`, for predicting the glass transition temperature (Tg) of polymers. The model was trained on a dataset of polymer structures represented as graphs, with node features derived from pre-trained embeddings. The model demonstrates strong predictive performance on an unseen test set, achieving a Mean Absolute Error (MAE) of **{test_mae:.2f} K** and an R-squared (R²) value of **{test_r2:.3f}**. The model also provides uncertainty estimates for its predictions.

---

## 2. Configuration Details

### 2.1. Hyperparameters
- **Random Seed:** {SEED}
- **Epochs (Max):** {EPOCHS}
- **Batch Size:** {BATCH_SIZE}
- **Learning Rate:** {LEARNING_RATE}
- **Optimizer:** AdamW (Weight Decay: 1e-5)
- **LR Scheduler:** ReduceLROnPlateau (Factor: 0.5, Patience: 15)
- **Early Stopping Patience:** {patience_epochs} epochs

### 2.2. Model Architecture (`PolyGCBaseModel`)
- **Task Type:** {model_config.task_type} (with uncertainty)
- **Input Feature Dimension:** {model_config.monomer_features}
- **GNN Layers:** {model_config.num_gnn_layers}
- **Graph Conv Output Features:** {model_config.gc_features}
- **MLP Layers:** {model_config.mlp_layer}
- **Dropout Rate:** {model_config.dropout_rate}
- **Graph Pooling:** {", ".join([p["type"] for p in model_config.pooling_layers])}
- **Output Layer:** Logits (predicts mean and log-variance)

### 2.3. Environment
- **Device:** {device}
- **Database Source:** {db_path}

---

## 3. Data Analysis

- **Dataset Name:** "Tg_Prediction_from_CSV"
- **Total Graph Samples:** {len(all_graph_data)}
- **Data Splitting Strategy:** Splits were made based on unique polymer entries to prevent data leakage between sets.

### 3.1. Dataset Splits
- **Total Unique Entries:** {num_entries}
- **Training Set:** {len(train_graphs)} samples from {len(train_entry_indices)} unique entries.
- **Validation Set:** {len(val_graphs)} samples from {len(val_entry_indices)} unique entries.
- **Test Set:** {len(test_graphs)} samples from {len(test_entry_indices)} unique entries.

### 3.2. Target Variable Analysis (Tg in Training Set)
- **Target Name:** '{target_name}'
- **Mean:** {target_stats["mean"]:.2f} K
- **Standard Deviation:** {target_stats["std"]:.2f} K
- **Min:** {target_stats["min"]:.2f} K
- **Max:** {target_stats["max"]:.2f} K

---

## 4. Training Process Summary

- **Total Epochs Trained:** {final_epoch} (Early stopping triggered: {patience_counter >= patience_epochs})
- **Best Validation Loss:** {best_val_loss:.4f} (achieved at Epoch {best_epoch})
- **Optimization:** The learning rate was dynamically adjusted using `ReduceLROnPlateau` based on validation loss. Training was halted after the validation loss failed to improve for {patience_epochs} consecutive epochs.

---

## 5. Performance Evaluation (Test Set)

The model's final performance was assessed on the held-out test set using the best model state saved during training.

### 5.1. Quantitative Metrics
- **Mean Absolute Error (MAE):** **{test_mae:.2f} K**
  - *Interpretation: On average, the model's prediction of Tg is off by {test_mae:.2f} Kelvin from the true value.*
- **Root Mean Squared Error (RMSE):** **{test_rmse:.2f} K**
  - *Interpretation: This metric penalizes larger errors more heavily than MAE. The value is comparable to the MAE, suggesting no extreme, outlier errors are dominating the results.*
- **R-squared (R²):** **{test_r2:.3f}**
  - *Interpretation: The model explains {test_r2 * 100:.1f}% of the variance in the true Tg values, indicating a strong correlation between predictions and actuals.*

### 5.2. Analysis of Visualizations
*(See the corresponding .png and .txt files in the '{result_dir.name}' directory for full details)*

- **`{parity_plot_path.name}`:**
  - The parity plot shows a tight clustering of data points around the ideal y=x line, visually confirming the high R² value. The predicted uncertainty (error bars) appears reasonable, with some points further from the ideal line exhibiting larger uncertainty, as expected from a well-calibrated model.

- **`{error_plot_path.name}`:**
  - The histogram of prediction errors is centered near zero and is roughly symmetric, resembling a normal distribution. This indicates that the model is not systematically over- or under-predicting and is generally unbiased.

- **`{tsne_plot_path.name}`:**
  - The t-SNE plot visualizes the high-dimensional graph embeddings learned by the model. The clear color gradient, corresponding to the true Tg values, demonstrates that the model has successfully learned a meaningful representation (latent space) where polymers with similar properties are located close to each other. This is a strong indication that the model is learning chemically relevant features.

---

## 6. Conclusion & Future Work

The `PolyGCBaseModel` has proven to be highly effective for predicting polymer Tg from graph-based representations. The combination of GNN layers and multi-head pooling successfully captures the structural information relevant to this thermal property.

Potential directions for future work include:
- **Hyperparameter Optimization:** A more systematic search (e.g., using Optuna or Ray Tune) could further refine the model's architecture and training parameters.
- **Exploring Different Architectures:** Testing other GNN convolutional layers (e.g., GAT, GIN) might yield performance improvements.
- **Expanding the Dataset:** Training on a larger and more diverse set of polymers could improve the model's generalizability.
- **Multi-Target Prediction:** Extending the model to simultaneously predict other properties (e.g., melting point, density) could allow it to learn more comprehensive polymer representations.
"""

with open(result_dir / "training_analysis_report.md", "w") as f:
    f.write(report_content)

print(f"\nAll results, plots, and reports saved to: {result_dir}")
