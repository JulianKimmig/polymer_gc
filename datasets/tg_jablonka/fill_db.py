import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

# Import the necessary components from your polymer_gc library
from polymer_gc.data.dataset import (
    PgDatasetConfig,
    Dataset,
    DatasetItem,
    GraphDatasetItemLink,
)
from polymer_gc.data.basemodels.structures import SQLStructureModel
from polymer_gc.data.graphs import ReusableGraphEntry
from polymer_gc.data.database import SessionManager

# --- Configuration ---

# 1. Define file paths and names
DB_PATH = "database.db"  # Use the same database as the classification script
CSV_PATH = "LAMALAB_CURATED_Tg_structured_polymerclass.csv"  # The path to your data
DATASET_NAME = "Tg_Prediction_from_CSV"


# 3. Define the dataset configuration
#    This tells the database how to interpret the data we are about to add.
pg_dataset_config = PgDatasetConfig(
    # The intended featurization method for the GCN part of the model.
    embedding="PolyBERT",
    # For regression, we have one target value.
    targets=["Tg"],
    # We will use the rich pre-calculated features as the "contextual vector".
    # Not needed for a regression task.
    target_classes={},
    # These are not directly applicable as we are not generating ensembles.
    # Set to reasonable defaults.
    num_bins=0,
)


def main():
    """
    Main function to read the CSV and populate the database.
    """
    print("--- Starting Tg Prediction Dataset Generation ---")

    # Load the CSV data into a pandas DataFrame
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Successfully loaded {CSV_PATH}. Found {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: The file {CSV_PATH} was not found.")
        return

    # Use the SessionManager to handle database connections
    with SessionManager(DB_PATH) as session:
        # Create or retrieve the main Dataset object
        dataset = Dataset.get_or_create(
            name=DATASET_NAME, set_kwargs=dict(config=pg_dataset_config)
        )

        general_graph = ReusableGraphEntry.get_or_create(
            name="tg_jablonka_general_graph",
            new_kwargs=dict(
                nodes=["A"] * 10,
                edges=[(i, i + 1) for i in range(9)],
                n_nodes=10,
                n_edges=9,
            ),
        )
        print(f"Using dataset '{DATASET_NAME}' (ID: {dataset.id})")

        # making structs
        index_to_structure = {}
        for index, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Processing polymers"
        ):
            psmiles = row.get("PSMILES")
            if pd.isna(psmiles) or not isinstance(psmiles, str) or not psmiles:
                print(f"Skipping row {index}: Invalid or missing PSMILES.")
                continue
            structure = SQLStructureModel.get_or_create(smiles=psmiles, commit=False)
            index_to_structure[index] = structure
        session.commit()

        # Iterate over each row in the DataFrame to create database entries
        for index, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Processing polymers"
        ):
            if index not in index_to_structure:
                # no structure for this index
                continue
            structure = index_to_structure[index]

            tg_values = row.get("meta.tg_values")
            # if tg values is not nan or empty string parse it as json
            if pd.isna(tg_values) or tg_values == "":
                tg_values = [row.get("labels.Exp_Tg(K)")]
            else:
                tg_values = json.loads(tg_values)

            for i, tg_value in reversed(list(enumerate(tg_values))):
                if pd.isna(tg_value):
                    tg_values.pop(i)

            if len(tg_values) == 0:
                print(f"Skipping row {index}: No Tg values.")
                continue

            # --- Database Object Creation ---

            for tg_value in tg_values:
                # 2. Create the DatasetItem, which links everything together.
                #    This is the main entry for one experimental data point.
                item = DatasetItem.get_or_create(
                    dataset=dataset,
                    entry_type="Tg_Prediction_Entry",
                    # Link to the polymer structure
                    structure_map={"A": structure.id},
                    # Set the regression target
                    targets={"Tg": float(tg_value)},
                    mn=0,
                    mw=0,
                    commit=False,
                )
                GraphDatasetItemLink.link(item, general_graph)

            # Commit changes to the database periodically for efficiency
            if (index + 1) % 500 == 0:
                session.commit()

        SQLStructureModel.batch_get_embedding(
            list(index_to_structure.values()), "PolyBERT"
        )
        # Final commit to save any remaining entries
        session.commit()
        print("--- Database population complete. ---")


if __name__ == "__main__":
    main()
