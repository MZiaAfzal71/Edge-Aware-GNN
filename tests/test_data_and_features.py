import numpy as np

from edge_aware_gnn.data import canonicalize_smiles, make_split
from edge_aware_gnn.features import BOND_FEATURE_DIM, MolecularFeatureTransformer, bond_features

SMILES = [
    "c1ccccc1",
    "Cc1ccccc1",
    "CCc1ccccc1",
    "c1ccncc1",
    "Cc1ccncc1",
    "C1CCCCC1",
    "CC1CCCCC1",
    "c1ccoc1",
    "Cc1ccoc1",
    "c1ccsc1",
    "Cc1ccsc1",
    "CCO",
]


def test_canonicalization_and_scaffold_split_are_disjoint():
    canonical = [canonicalize_smiles(value) for value in SMILES]
    assert all(canonical)
    split = make_split(
        canonical, strategy="scaffold", seed=2026, fractions=(0.4, 0.2, 0.2, 0.2)
    )
    scaffold_sets = {
        name: set(split.loc[split["partition"] == name, "scaffold"])
        for name in ("train", "validation", "calibration", "test")
    }
    assert scaffold_sets["train"].isdisjoint(scaffold_sets["validation"])
    assert scaffold_sets["train"].isdisjoint(scaffold_sets["test"])
    assert scaffold_sets["validation"].isdisjoint(scaffold_sets["test"])
    assert scaffold_sets["calibration"].isdisjoint(scaffold_sets["train"])
    assert scaffold_sets["calibration"].isdisjoint(scaffold_sets["validation"])
    assert scaffold_sets["calibration"].isdisjoint(scaffold_sets["test"])


def test_train_fitted_descriptor_and_fingerprint_shapes_match():
    transformer = MolecularFeatureTransformer("rdkit+morgan", morgan_bits=64)
    train = transformer.fit_transform(SMILES[:9])
    test = transformer.transform(SMILES[9:])
    assert train.shape[0] == 9
    assert test.shape[0] == 3
    assert train.shape[1] == test.shape[1] == len(transformer.feature_names_)
    assert np.isfinite(train).all()
    assert np.isfinite(test).all()


def test_bond_vector_dimension():
    from rdkit import Chem

    molecule = Chem.MolFromSmiles("C/C=C\\Cl")
    assert len(bond_features(molecule.GetBondWithIdx(1))) == BOND_FEATURE_DIM
