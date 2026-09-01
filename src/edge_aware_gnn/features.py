"""Train-only global features and chemically explicit graph featurization."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Atomic number, degree, formal charge, total H, hybridization, chirality,
# valence, radical electrons, aromaticity, ring membership.
ATOM_VOCAB_SIZES = (119, 11, 11, 9, 7, 5, 13, 5, 2, 2)
BOND_FEATURE_DIM = 14  # 5 bond types + 7 stereo states + conjugation + ring


def _molecule(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def atom_categories(atom) -> list[int]:
    from rdkit import Chem

    hybridization_map = {
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.SP2: 3,
        Chem.rdchem.HybridizationType.SP3: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6,
    }
    raw = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        int(np.clip(atom.GetFormalCharge(), -5, 5)) + 5,
        atom.GetTotalNumHs(),
        hybridization_map.get(atom.GetHybridization(), 0),
        int(atom.GetChiralTag()),
        atom.GetTotalValence(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]
    return [int(np.clip(value, 0, size - 1)) for value, size in zip(raw, ATOM_VOCAB_SIZES)]


def _one_hot(index: int, size: int) -> list[float]:
    result = [0.0] * size
    result[int(np.clip(index, 0, size - 1))] = 1.0
    return result


def bond_features(bond) -> list[float]:
    from rdkit import Chem

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    bond_type = bond_type_map.get(bond.GetBondType(), 4)
    stereo = int(np.clip(int(bond.GetStereo()), 0, 6))
    return _one_hot(bond_type, 5) + _one_hot(stereo, 7) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(
    smiles: str,
    *,
    target: float,
    global_features: np.ndarray | None = None,
):
    """Convert one molecule to a PyG graph, including safe zero-bond shapes."""
    import torch
    from torch_geometric.data import Data

    mol = _molecule(smiles)
    x = torch.tensor([atom_categories(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    edges: list[list[int]] = []
    attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        attr = bond_features(bond)
        edges.extend(([i, j], [j, i]))
        attrs.extend((attr, attr))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target], dtype=torch.float32),
        smiles=smiles,
    )
    if global_features is not None:
        data.global_features = torch.as_tensor(global_features, dtype=torch.float32).reshape(1, -1)
    return data


class MolecularFeatureTransformer:
    """RDKit descriptor cleaning plus optional Morgan fingerprints.

    Missingness filtering, imputation, variance filtering, correlation pruning,
    and scaling are learned from training molecules only.
    """

    def __init__(
        self,
        feature_set: str,
        *,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        max_missing_fraction: float = 0.10,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        standardized_clip: float = 10.0,
    ):
        if feature_set not in {"none", "rdkit", "morgan", "rdkit+morgan"}:
            raise ValueError(f"Unknown feature set {feature_set!r}")
        if standardized_clip <= 0:
            raise ValueError("standardized_clip must be positive")
        self.feature_set = feature_set
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.max_missing_fraction = max_missing_fraction
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.standardized_clip = float(standardized_clip)
        self.descriptor_columns_: list[str] = []
        self.correlation_keep_: np.ndarray | None = None
        self.imputer_: SimpleImputer | None = None
        self.variance_: VarianceThreshold | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []

    @property
    def uses_rdkit(self) -> bool:
        return "rdkit" in self.feature_set

    @property
    def uses_morgan(self) -> bool:
        return "morgan" in self.feature_set

    def _descriptors(self, smiles: Iterable[str]) -> pd.DataFrame:
        from rdkit.Chem import Descriptors

        descriptor_list = Descriptors.descList
        records: list[list[float]] = []
        for value in smiles:
            mol = _molecule(value)
            row: list[float] = []
            for _, function in descriptor_list:
                try:
                    result = float(function(mol))
                    row.append(result if np.isfinite(result) else np.nan)
                except (OverflowError, RuntimeError, TypeError, ValueError):
                    row.append(np.nan)
            records.append(row)
        return pd.DataFrame(records, columns=[name for name, _ in descriptor_list])

    def _morgan(self, smiles: Iterable[str]) -> np.ndarray:
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius, fpSize=self.morgan_bits
        )
        rows: list[np.ndarray] = []
        for value in smiles:
            rows.append(generator.GetFingerprintAsNumPy(_molecule(value)).astype(np.float32))
        return np.vstack(rows)

    def fit(self, smiles: Iterable[str]) -> MolecularFeatureTransformer:
        values = list(smiles)
        if self.uses_rdkit:
            frame = self._descriptors(values)
            missing = frame.isna().mean()
            self.descriptor_columns_ = missing[missing <= self.max_missing_fraction].index.tolist()
            if not self.descriptor_columns_:
                raise ValueError("No RDKit descriptors survived missingness filtering")
            selected = frame[self.descriptor_columns_].to_numpy(dtype=float)
            self.imputer_ = SimpleImputer(strategy="median")
            imputed = self.imputer_.fit_transform(selected)
            self.variance_ = VarianceThreshold(threshold=self.variance_threshold)
            variable = self.variance_.fit_transform(imputed)
            if variable.shape[1] == 0:
                raise ValueError("No descriptors survived variance filtering")
            correlation = np.abs(np.corrcoef(variable, rowvar=False))
            upper = np.triu(correlation, k=1)
            self.correlation_keep_ = np.flatnonzero(~(upper > self.correlation_threshold).any(axis=0))
            pruned = variable[:, self.correlation_keep_]
            self.scaler_ = StandardScaler().fit(pruned)
            variable_names = np.asarray(self.descriptor_columns_)[self.variance_.get_support()]
            self.feature_names_.extend(variable_names[self.correlation_keep_].tolist())
        if self.uses_morgan:
            self.feature_names_.extend([f"morgan_{i}" for i in range(self.morgan_bits)])
        return self

    def transform(self, smiles: Iterable[str]) -> np.ndarray:
        values = list(smiles)
        blocks: list[np.ndarray] = []
        if self.uses_rdkit:
            if any(value is None for value in (self.imputer_, self.variance_, self.scaler_, self.correlation_keep_)):
                raise RuntimeError("Feature transformer has not been fitted")
            frame = self._descriptors(values).reindex(columns=self.descriptor_columns_)
            imputed = self.imputer_.transform(frame.to_numpy(dtype=float))
            variable = self.variance_.transform(imputed)
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = self.scaler_.transform(variable[:, self.correlation_keep_])
            scaled = np.nan_to_num(
                scaled,
                nan=0.0,
                posinf=self.standardized_clip,
                neginf=-self.standardized_clip,
            )
            scaled = np.clip(scaled, -self.standardized_clip, self.standardized_clip)
            blocks.append(scaled.astype(np.float32))
        if self.uses_morgan:
            blocks.append(self._morgan(values))
        if not blocks:
            return np.empty((len(values), 0), dtype=np.float32)
        result = np.concatenate(blocks, axis=1)
        if not np.isfinite(result).all():
            bad = int((~np.isfinite(result)).sum())
            raise FloatingPointError(f"Molecular feature matrix contains {bad} non-finite values")
        return result

    def fit_transform(self, smiles: Iterable[str]) -> np.ndarray:
        values = list(smiles)
        return self.fit(values).transform(values)


def build_feature_transformer(feature_set: str, options: dict) -> MolecularFeatureTransformer:
    return MolecularFeatureTransformer(feature_set, **options)


def make_graph_dataset(
    frame: pd.DataFrame,
    targets: np.ndarray,
    global_features: np.ndarray | None,
):
    from torch.utils.data import Dataset

    graphs = [
        smiles_to_graph(
            row.smiles,
            target=float(targets[index]),
            global_features=None if global_features is None else global_features[index],
        )
        for index, row in enumerate(frame.itertuples(index=False))
    ]

    class _ListDataset(Dataset):
        def __init__(self, values):
            self._values = values

        def __len__(self):
            return len(self._values)

        def __getitem__(self, index):
            return self._values[index]

    return _ListDataset(graphs)
