from edge_aware_gnn.validation import validate_repository


def test_repository_structure_and_completed_results_are_valid():
    report = validate_repository(".")
    assert report["status"] == "ok"
    assert report["expected_runs"] == 480
    assert report["metric_rows"] == 1920
    assert report["prediction_files"] == 480
    assert report["applicability_domain_rows"] == 232
