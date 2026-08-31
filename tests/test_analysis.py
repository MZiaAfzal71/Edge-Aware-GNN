import pandas as pd

from edge_aware_gnn.analysis import nearest_training_tanimoto


def test_nearest_training_similarity_is_defined_only_for_holdout_molecules():
    frame = pd.DataFrame(
        {
            "molecule_id": ["a", "b", "c", "d"],
            "smiles": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "CCO"],
            "partition": ["train", "train", "validation", "test"],
        }
    )
    similarity = nearest_training_tanimoto(frame, bits=128)
    assert similarity.iloc[:2].isna().all()
    assert similarity.iloc[2:].between(0, 1).all()
    assert similarity.iloc[2] > similarity.iloc[3]

