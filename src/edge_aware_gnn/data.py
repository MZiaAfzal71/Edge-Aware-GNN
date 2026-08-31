"""MoleculeNet loading, curation, and leakage-safe data splitting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_REGISTRY = {
    "esol": {
        "pyg_name": "ESOL",
        "property": "aqueous_solubility",
        "units": "log10(mol/L)",
        "recommended_primary_split": "scaffold",
    },
    "freesolv": {
        "pyg_name": "FreeSolv",
        "property": "hydration_free_energy",
        "units": "kcal/mol",
        "recommended_primary_split": "random",
    },
    "lipophilicity": {
        "pyg_name": "Lipo",
        "property": "octanol_water_distribution_coefficient",
        "units": "logD (pH 7.4)",
        "recommended_primary_split": "scaffold",
    },
}
PARTITIONS = ("train", "validation", "calibration", "test")


@dataclass(frozen=True)
class DatasetAudit:
    dataset: str
    raw_rows: int
    invalid_smiles: int
    non_finite_targets: int
    duplicate_rows: int
    curated_rows: int
    target_min: float
    target_max: float
    target_mean: float
    target_std: float


def canonicalize_smiles(smiles: str) -> str | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_moleculenet(dataset_name: str, data_root: str | Path) -> tuple[pd.DataFrame, DatasetAudit]:
    """Load a PyG MoleculeNet dataset and curate canonical-SMILES duplicates.

    Duplicate structures are collapsed with the median target. The target range
    among duplicates is retained so disagreements can be audited rather than hidden.
    """
    key = dataset_name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {dataset_name!r}")

    from torch_geometric.datasets import MoleculeNet

    root = Path(data_root) / key
    dataset = MoleculeNet(root=str(root), name=DATASET_REGISTRY[key]["pyg_name"])
    records: list[dict[str, float | str | int]] = []
    invalid_smiles = 0
    non_finite_targets = 0
    for raw_index, item in enumerate(dataset):
        smiles = str(item.smiles)
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            invalid_smiles += 1
            continue
        target = float(item.y.reshape(-1)[0].item())
        if not np.isfinite(target):
            non_finite_targets += 1
            continue
        records.append(
            {"raw_index": raw_index, "raw_smiles": smiles, "smiles": canonical, "target": target}
        )

    raw = pd.DataFrame.from_records(records)
    grouped = raw.groupby("smiles", sort=True, as_index=False).agg(
        target=("target", "median"),
        duplicate_count=("target", "size"),
        duplicate_target_min=("target", "min"),
        duplicate_target_max=("target", "max"),
    )
    grouped["molecule_id"] = grouped["smiles"].map(
        lambda value: sha256(f"{key}:{value}".encode()).hexdigest()[:16]
    )
    grouped.insert(0, "dataset", key)
    grouped["property"] = DATASET_REGISTRY[key]["property"]
    grouped["units"] = DATASET_REGISTRY[key]["units"]

    audit = DatasetAudit(
        dataset=key,
        raw_rows=len(dataset),
        invalid_smiles=invalid_smiles,
        non_finite_targets=non_finite_targets,
        duplicate_rows=int((grouped["duplicate_count"] - 1).sum()),
        curated_rows=len(grouped),
        target_min=float(grouped["target"].min()),
        target_max=float(grouped["target"].max()),
        target_mean=float(grouped["target"].mean()),
        target_std=float(grouped["target"].std(ddof=1)),
    )
    return grouped, audit


def scaffold_from_smiles(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=False
    )


def _partition_sizes(n_items: int, fractions: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(fractions), dtype=float)
    if len(values) != len(PARTITIONS) or not np.isclose(values.sum(), 1.0):
        raise ValueError(f"fractions must be {len(PARTITIONS)} values summing to one")
    raw = values * n_items
    sizes = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - sizes))[: n_items - sizes.sum()]:
        sizes[index] += 1
    return sizes


def make_split(
    smiles: Iterable[str],
    *,
    strategy: str,
    seed: int,
    fractions: Iterable[float] = (0.70, 0.10, 0.10, 0.10),
) -> pd.DataFrame:
    """Create explicit train/validation/calibration/test assignments without targets."""
    smiles_values = list(smiles)
    n_items = len(smiles_values)
    if n_items < 3:
        raise ValueError("At least three molecules are required")
    target_sizes = _partition_sizes(n_items, fractions)
    rng = np.random.default_rng(seed)
    partitions = np.empty(n_items, dtype=object)
    scaffold_values: list[str | None] = [None] * n_items

    if strategy == "random":
        order = rng.permutation(n_items)
        start = 0
        for label, size in zip(PARTITIONS, target_sizes):
            partitions[order[start : start + size]] = label
            start += size
    elif strategy == "scaffold":
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(smiles_values):
            scaffold = scaffold_from_smiles(value)
            scaffold_values[index] = scaffold
            groups.setdefault(scaffold, []).append(index)

        # Randomizing before a stable size sort changes only ties, producing
        # repeated target-independent scaffold partitions across seeds.
        grouped_indices = list(groups.values())
        rng.shuffle(grouped_indices)
        grouped_indices.sort(key=len, reverse=True)
        assigned: list[list[int]] = [[] for _ in PARTITIONS]
        for group in grouped_indices:
            remaining = target_sizes - np.asarray([len(values) for values in assigned])
            eligible = np.flatnonzero(remaining > 0)
            destination = int(eligible[np.argmax(remaining[eligible])]) if len(eligible) else int(
                np.argmin(np.asarray([len(values) for values in assigned]) / target_sizes)
            )
            assigned[destination].extend(group)
        for label, indices in zip(PARTITIONS, assigned):
            partitions[indices] = label
    else:
        raise ValueError("strategy must be 'random' or 'scaffold'")

    result = pd.DataFrame(
        {
            "row_index": np.arange(n_items),
            "partition": partitions,
            "scaffold": scaffold_values,
        }
    )
    validate_split(result, strategy=strategy)
    return result


def validate_split(split: pd.DataFrame, *, strategy: str) -> None:
    expected = set(PARTITIONS)
    observed = set(split["partition"])
    if observed != expected:
        raise ValueError(f"Split must contain {expected}; observed {observed}")
    if split["row_index"].duplicated().any():
        raise ValueError("A molecule appears in more than one partition")
    if strategy == "scaffold":
        sets = {
            name: set(split.loc[split["partition"] == name, "scaffold"])
            for name in expected
        }
        pairs = (
            (PARTITIONS[i], PARTITIONS[j])
            for i in range(len(PARTITIONS))
            for j in range(i + 1, len(PARTITIONS))
        )
        if any(sets[a] & sets[b] for a, b in pairs):
            raise ValueError("A Bemis-Murcko scaffold crosses partitions")


def audit_as_dict(audit: DatasetAudit) -> dict[str, int | float | str]:
    return asdict(audit)
