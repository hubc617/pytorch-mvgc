import numpy as np
from pytorch_mvgc import compute_mvgc


def test_causal_direction():
    rng = np.random.default_rng(42)
    x = np.zeros((3, 2000))
    noise = rng.normal(size=x.shape)
    for t in range(1, x.shape[1]):
        x[0, t] = 0.8 * x[0, t-1] + noise[0, t]
        x[1, t] = 0.5 * x[1, t-1] + 0.5 * x[0, t-1] + noise[1, t]
        x[2, t] = 0.7 * x[2, t-1] + noise[2, t]
    result = compute_mvgc(x, p=1, device='cpu')
    assert result[1, 0] > 0.05
    assert result[0, 1] < 0.05
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(np.diag(result), 0)
