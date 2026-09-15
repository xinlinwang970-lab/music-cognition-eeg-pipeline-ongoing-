import numpy as np

from scripts.analysis.ds002721_features import beta_log_power


def test_20hz_sine_exceeds_11hz_in_beta_band():
    fs = 1000.0
    t = np.arange(0, 20.0, 1.0 / fs)
    beta20 = beta_log_power(np.sin(2 * np.pi * 20.0 * t), fs)
    alpha11 = beta_log_power(np.sin(2 * np.pi * 11.0 * t), fs)
    assert beta20 > alpha11
