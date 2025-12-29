import numpy as np
import pytest
from nig_model import NormalInverseGammaState

def test_update_increments_n(state: NormalInverseGammaState, samples):
    for x in samples:
        state.update(x)
    
    assert state.n == len(samples)

def test_update_accumulates_sum_x_and_sum_x2(state: NormalInverseGammaState, samples):
    for x in samples:
        state.update(x)
    
    assert state.sum_x == pytest.approx(sum(samples), abs=1e-12)
    assert state.sum_x2 == pytest.approx(sum(samples**2), abs=1e-12)

def test_mean_matches_batch_mean(state, samples):
    for x in samples:
        state.update(x)
    
    assert state.mean() == pytest.approx(float(np.mean(samples)), abs=1e-12)

def test_sse_matches_batch_sse(state: NormalInverseGammaState, samples):
    for x in samples:
        state.update(x)
    
    mean_batch = np.mean(samples)
    sse_batch = np.sum((samples - mean_batch)**2)

    # Guardails
    assert state.sse() >= -1e-12
    assert state.sse() == pytest.approx(sse_batch, abs=1e-12)
    