from scripts.analysis.ds002721_features import FEATURE_SPEC_ID, PREPROCESSING_ID


def test_required_schema_identifiers_are_frozen():
    assert FEATURE_SPEC_ID == "frontal_beta_v1"
    assert PREPROCESSING_ID == "raw_edf_no_artifact_rejection_trial_level_v1"
