import pandas as pd
from tqdm import tqdm
import json
from typing import Optional
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
from polymer_gc.sec import SimSEC
from polymer_gc.data.sec import SECEntry
from pathlib import Path
# --- Configuration ---

# 1. Define file paths and names
CSV_PATH = Path(__file__).parent / "LAMALAB_CURATED_Tg_structured_polymerclass.csv"  # The path to your data

default_pg_dataset_config = PgDatasetConfig(
    targets=["Tg"],
)

def populate(
        db_path:Path,
        dataset_name="tg_jablonka",
        virtual_mn=50_000,
        virtual_pdi=1.3,	
        pg_dataset_config:Optional[PgDatasetConfig]=None
):
    """
    Main function to read the CSV and populate the database.
    """
    print("--- Starting Tg Prediction Dataset Generation ---")

    virtual_mw = virtual_mn * virtual_pdi

    pg_dataset_config = pg_dataset_config or default_pg_dataset_config
    pg_dataset_config.targets = default_pg_dataset_config.targets
    pg_dataset_config.target_classes = None
    # Load the CSV data into a pandas DataFrame
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Successfully loaded {CSV_PATH}. Found {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: The file {CSV_PATH} was not found.")
        return

    # Use the SessionManager to handle database connections
    with SessionManager(db_path) as session:
        # Create or retrieve the main Dataset object
        dataset = Dataset.get_or_create(
            name=dataset_name, set_kwargs=dict(config=pg_dataset_config)
        )
        assert dataset.dict_config is not None, "Dataset dict_config is None"
        assert dataset.config is not None, "Dataset config is None"

        
        general_sec = SECEntry.all()
        if not general_sec:
            sec = SimSEC.from_mn_mw(Mw=virtual_mw, Mn=virtual_mn)
            general_sec = SECEntry.from_sec(sec)
        else:
            general_sec = general_sec[0]

        # making structs
        index_to_structure = {}
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="making structs"):
            psmiles = row.get("PSMILES")
            if pd.isna(psmiles) or not isinstance(psmiles, str) or not psmiles:
                print(f"Skipping row {index}: Invalid or missing PSMILES.")
                continue
            structure = SQLStructureModel.get_or_create(smiles=psmiles, commit=False)
            index_to_structure[index] = structure
        session.commit()

        general_graph = ReusableGraphEntry.get_or_create(
            name="tg_jablonka_general_graph",
            new_kwargs=dict(
                nodes=["A"] * 10,
                edges=[(i, i + 1) for i in range(9)],
                n_nodes=10,
                n_edges=9,
            ),
        )
        print(f"Using dataset '{dataset_name}' (ID: {dataset.id})")

        # making structs

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
                    sec_id=general_sec.id,
                )
                GraphDatasetItemLink.link(item, general_graph)

            # Commit changes to the database periodically for efficiency
            if (index + 1) % 500 == 0:
                session.commit()

        SQLStructureModel.batch_get_embedding(
            list(index_to_structure.values()), dataset.config.embedding
        )
        # Final commit to save any remaining entries
        session.commit()
        print("--- Database population complete. ---")


if __name__ == "__main__":
    populate(
        db_path=Path(__file__).parent / "database_jb.db",
        dataset_name="Tg_Prediction_from_CSV_jb",
        
    )
