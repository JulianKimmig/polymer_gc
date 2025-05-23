import umap
import torch
import matplotlib.pyplot as plt
import polymer_gc
import os
import numpy as np
from tqdm import tqdm
from typing import Optional
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.preprocessing import StandardScaler
from polymer_gc.model.base import PolyGCBaseModel


def plot_embeddings_map(
    source_dataset: polymer_gc.dataset.Dataset,
    pg_dataset_config: polymer_gc.dataset.PgDatasetConfig,
    path: str,
    overwrite: bool = False,
):
    if overwrite:
        targets_to_make = [k for k in pg_dataset_config.targets]
    else:
        targets_to_make = [
            k
            for k in pg_dataset_config.targets
            if not os.path.exists(
                f"{path}/umap_{source_dataset.name}_{pg_dataset_config.embedding}_{k}.png"
            )
        ]

    if len(targets_to_make) == 0:
        print("All targets already plotted. Exiting...")
        return
    unique_monomers = source_dataset.unique_monomers
    if len(unique_monomers) == 0:
        raise ValueError("No unique monomers found in the dataset.")

    # get or create embeddings
    print("Loading or creating embeddings...")
    unique_monomers[0].add_batch_embeddings(
        unique_monomers,
        pg_dataset_config.embedding,
        values=[
            monomer.embeddings.get(pg_dataset_config.embedding, None)
            for monomer in unique_monomers
        ],
    )

    targets = {k: [] for k in targets_to_make}
    embeddings = []
    for entry in source_dataset.items:
        entrytargets = {k: getattr(entry, k) for k in targets_to_make}
        for monomer in entry.monomers:
            embeddings.append(
                torch.Tensor(
                    monomer.embeddings[pg_dataset_config.embedding].value
                ).float()
            )
            for k, v in entrytargets.items():
                targets[k].append(v)

    embeddings = torch.stack(embeddings, dim=0)

    print("Calculating UMAP...")
    reducer = umap.UMAP()
    embeddings = StandardScaler().fit_transform(embeddings)
    embedding = reducer.fit_transform(embeddings)

    # calcualte marker size based on number of unique monomers
    num_unique_monomers = len(unique_monomers)
    best_marker_size = 1000 / np.sqrt(num_unique_monomers)

    if not os.path.exists(path):
        os.makedirs(path)
    for k in targets.keys():
        print(f"Plotting for {k}...")
        plt.figure()
        plt.title(
            f"UMAP projection of {pg_dataset_config.embedding} embeddings for {source_dataset.name}"
        )
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=targets[k],
            marker=".",
            s=best_marker_size,
            alpha=1,
            cmap="viridis",
        )
        plt.gca().set_aspect("equal", "datalim")
        plt.colorbar()
        # set colorbar name
        plt.gca().collections[0].colorbar.set_label(k)
        # add axis labels
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.savefig(
            f"{path}/umap_{source_dataset.name}_{pg_dataset_config.embedding}_{k}.png"
        )
        plt.close()


def _model_outs_to_target_keys(
    model: PolyGCBaseModel,
    pg_dataset_config: polymer_gc.dataset.PgDatasetConfig,
):
    if model.config.num_target_properties == len(pg_dataset_config.targets):
        return [k for k in pg_dataset_config.targets]

    return list(range(model.config.num_target_properties))


def plot_model_embeddings(
    model: PolyGCBaseModel,
    pg_dataset_config: polymer_gc.dataset.PgDatasetConfig,
    path: str,
    overwrite: bool = False,
    train_loader: Optional[PyGDataLoader] = None,
    test_loader: Optional[PyGDataLoader] = None,
    val_loader: Optional[PyGDataLoader] = None,
):
    target_keys = _model_outs_to_target_keys(model, pg_dataset_config)
    if overwrite:
        targets_to_make = target_keys
    else:
        targets_to_make = [
            i
            for i in target_keys
            if not os.path.exists(f"{path}/umap_model_target_{i}.png")
        ]

    if len(targets_to_make) == 0:
        print("All targets already plotted. Exiting...")
        return
    ofun = model.readout.forward

    pre_readouts = []

    def readout(x):
        pre_readouts.append(x.detach().cpu())
        return ofun(x)

    model.readout.forward = readout
    print("Collect embeddings from model...")
    try:
        ys = []
        ypreds = []
        for loader in [train_loader, test_loader, val_loader]:
            if loader is None:
                continue

            for batch in tqdm(
                loader,
                desc="Collect embeddings",
                total=len(loader),
            ):
                batch = batch.to("cuda")
                pred = model.predict(batch)
                if model.config.logits_output:
                    ypred = pred[0].detach()
                else:
                    ypred = pred.detach()
                ys.append(batch.y.cpu())
                ypreds.append(ypred.cpu())

    finally:
        model.readout.forward = ofun
    pre_readouts = torch.cat(pre_readouts).numpy()
    ys = torch.cat(ys).cpu().numpy()
    ypreds = torch.cat(ypreds).cpu().numpy()
    print("Calculating UMAP...")
    reducer = umap.UMAP()
    pre_readouts = StandardScaler().fit_transform(pre_readouts)
    embedding = reducer.fit_transform(pre_readouts)
    # calcualte marker size based on number of unique monomers

    best_marker_size = 1000 / np.sqrt(pre_readouts.shape[0])

    if not os.path.exists(path):
        os.makedirs(path)
    for i, k in enumerate(target_keys):
        print(f"Plotting for {k}...")
        plt.figure()
        plt.title(
            f"UMAP projection of {pg_dataset_config.embedding} embeddings for {model.config.model}"
        )
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=ys[:, i],
            marker=".",
            s=best_marker_size,
            alpha=1,
            cmap="viridis",
        )
        plt.gca().set_aspect("equal", "datalim")
        plt.colorbar()
        # set colorbar name
        plt.gca().collections[0].colorbar.set_label(f"feature {k}")
        # add axis labels
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.savefig(f"{path}/umap_model_target_{k}.png")
        plt.close()


def plot_true_vs_pred(
    model: PolyGCBaseModel,
    pg_dataset_config: polymer_gc.dataset.PgDatasetConfig,
    path: str,
    train_loader: Optional[PyGDataLoader] = None,
    test_loader: Optional[PyGDataLoader] = None,
    val_loader: Optional[PyGDataLoader] = None,
):
    target_keys = _model_outs_to_target_keys(model, pg_dataset_config)
    if not os.path.exists(path):
        os.makedirs(path)

    fys = {}
    fypreds = {}
    fystds = {}
    for name, loader in [
        ("train", train_loader),
        ("test", test_loader),
        ("val", val_loader),
    ]:
        if loader is None:
            continue

        ys = []
        ypreds = []
        ystds = []

        for batch in tqdm(
            loader,
            desc="Collect embeddings",
            total=len(loader),
        ):
            batch = batch.to("cuda")
            pred = model.predict(batch)
            if model.config.logits_output:
                ypred = pred[0].detach()
                ystd = pred[1].detach()
                ystds.append(ystd.cpu())
            else:
                ypred = pred.detach()
            ys.append(batch.y.cpu())
            ypreds.append(ypred.cpu())

        ys = torch.cat(ys).cpu().numpy()
        ypreds = torch.cat(ypreds).cpu().numpy()
        if model.config.logits_output:
            ystds = torch.exp(torch.cat(ystds)).sqrt().cpu().numpy()

        fys[name] = ys
        fypreds[name] = ypreds
        if model.config.logits_output:
            fystds[name] = ystds

    subset_keys = list(fys.keys())

    for col in [fys, fypreds, fystds]:
        fall = []
        for k in col.keys():
            fall.append(col[k])
        if len(fall) == 0:
            continue
        col["all"] = np.concatenate(fall, axis=0)

    for sk in subset_keys + ["all"]:
        for i, k in enumerate(target_keys):
            _y, _yp = fys[sk][:, i], fypreds[sk][:, i]
            # hexin
            plt.figure()
            plt.hexbin(_y, _yp)
            plt.xlabel(f"True {k}")
            plt.ylabel(f"Predicted {k}")
            plt.title(f"Hexbin plot of true vs predicted {k}")
            plt.colorbar()
            plt.gca().collections[0].colorbar.set_label("Counts")
            plt.savefig(f"{path}/hexbin_true_vs_pred_{k}_{sk}.png")
            plt.close()

            # scatter
            plt.figure()
            plt.scatter(
                _y,
                _yp,
                s=min(10, 1000 / np.sqrt(_y.shape[0])),
                alpha=0.5,
            )
            plt.xlabel(f"True {k}")
            plt.ylabel(f"Predicted {k}")
            plt.title(f"Scatter plot of true vs predicted {k}")

            rmse = np.sqrt(np.mean((_y - _yp) ** 2))
            rsquared = np.corrcoef(_y, _yp)[0, 1] ** 2
            mae = np.mean(np.abs(_y - _yp))

            # add information box to the plot
            textstr = f"RMSE: {rmse:.2f}\nR^2: {rsquared:.2f}\nMAE: {mae:.2f}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            # place a text box in upper left in axes coords
            plt.gca().text(
                0.05,
                0.95,
                textstr,
                transform=plt.gca().transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )
            plt.gca().set_aspect("equal", "datalim")

            plt.savefig(f"{path}/scatter_true_vs_pred_{k}_{sk}.png")
            plt.close()

            if model.config.logits_output:
                # plot each point as a gaussian with std
                plt.figure()
                plt.title(f"Gaussian plot of true vs predicted {k}")
                