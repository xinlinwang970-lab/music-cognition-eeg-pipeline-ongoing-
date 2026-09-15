import pandas as pd

from scripts.analysis.ds002721_features import subject_effects


def test_trial_table_has_one_final_beta_value_per_trial():
    df = pd.DataFrame(
        {
            "subject_id": ["sub-01"] * 5,
            "trial_id": [f"t{i}" for i in range(5)],
            "beta_log_power": [1, 2, 3, 4, 5],
            "frontal_roi_beta_log_power": [1, 2, 3, 4, 5],
            "energy": [1, 2, 3, 4, 5],
            "tension": [5, 4, 3, 2, 1],
            "pleasantness": [1, 1, 2, 2, 3],
        }
    )
    assert not df["trial_id"].duplicated().any()
    effects = subject_effects(df)
    assert effects.loc[0, "n_trials"] == 5
    assert effects.loc[0, "rho_energy_primary"] == 0.9999999999999999
