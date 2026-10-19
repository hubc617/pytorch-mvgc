from pathlib import Path
import numpy as np
import pytest
import torch
from fast_mvgc import compute_mvgc, compute_mvgc_batch, segment_data
from fast_mvgc.cpu_engine import compute_mvgc_cpu


def simulate(order=2, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 2400))
    for t in range(order, x.shape[1]):
        x[0, t] += 0.4 * x[0, t-1]
        x[1, t] += 0.3 * x[1, t-1] + 0.8 * x[0, t-order]
        x[2, t] += 0.3 * x[2, t-1]
    return x[:, 400:]


def reference(x, p, alpha):
    # Independent channel-major design and augmented least squares ridge fit.
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n, t = x.shape
    y = x[:, p:].T
    z = np.column_stack([x[c, p-k:t-k] for c in range(n) for k in range(1, p+1)])
    def sse(design):
        d = design.shape[1]
        a = np.vstack([design, np.sqrt(alpha) * np.eye(d)])
        b = np.vstack([y, np.zeros((d, n))])
        beta = np.linalg.lstsq(a, b, rcond=None)[0]
        return ((y - design @ beta)**2).sum(axis=0)
    full = sse(z)
    result = np.zeros((n, n))
    for source in range(n):
        reduced = sse(z[:, np.arange(n*p) // p != source])
        result[:, source] = np.maximum(np.log((reduced + 1e-12)/(full + 1e-12)), 0)
    np.fill_diagonal(result, 0)
    return result


def test_documented_namespace_is_shipped():
    import pytorch_mvgc
    assert Path(pytorch_mvgc.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])


@pytest.mark.parametrize('p', [1, 2, 3])
@pytest.mark.parametrize('alpha', [0.0, 0.01, 2.0])
def test_matches_independent_regression(p, alpha):
    x = simulate(order=p)
    expected = reference(x, p, alpha)
    np.testing.assert_allclose(compute_mvgc(x, p, alpha, 'cpu'), expected, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(compute_mvgc_cpu(x, p, alpha), expected, atol=1e-10, rtol=1e-8)
    assert expected[1, 0] > 0.3
    assert expected[0, 1] < 0.02


def test_batch_accepts_lists_and_negative_strides():
    x = np.stack([simulate(), simulate(seed=43)])[:, :, ::-1]
    expected = np.stack([reference(item, 2, 0.01) for item in x])
    for data in [x, x.tolist()]:
        np.testing.assert_allclose(compute_mvgc_batch(data, 2, device='cpu'), expected, atol=1e-10)


@pytest.mark.parametrize('data', [np.zeros((3, 40)), np.ones((3, 40)), np.tile(np.arange(40), (3, 1))])
@pytest.mark.parametrize('alpha', [0.0, 1e-20, 0.01])
def test_degenerate_inputs(data, alpha):
    expected = reference(data, 2, alpha)
    for result in [compute_mvgc(data, 2, alpha, 'cpu'), compute_mvgc_cpu(data, 2, alpha)]:
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, expected, atol=1e-9)


@pytest.mark.parametrize('p', [0, -1, 1.5, True, 40])
@pytest.mark.parametrize('engine', ['torch', 'numpy'])
def test_invalid_order(p, engine):
    with pytest.raises(ValueError, match='p|order'):
        if engine == 'torch':
            compute_mvgc(np.zeros((3, 40)), p, device='cpu')
        else:
            compute_mvgc_cpu(np.zeros((3, 40)), p)


@pytest.mark.parametrize('alpha', [-1, np.nan, np.inf, 'bad', True])
@pytest.mark.parametrize('engine', ['torch', 'numpy'])
def test_invalid_regularization(alpha, engine):
    with pytest.raises(ValueError, match='alpha'):
        if engine == 'torch':
            compute_mvgc(np.zeros((3, 40)), 1, alpha, 'cpu')
        else:
            compute_mvgc_cpu(np.zeros((3, 40)), 1, alpha)


@pytest.mark.parametrize('data', [np.empty((0, 3, 40)), np.empty((1, 0, 40)), np.zeros((3, 40)), np.full((1, 3, 40), np.nan), np.full((1, 3, 40), np.inf), np.ones((1, 3, 40), dtype=complex)])
def test_invalid_batch(data):
    with pytest.raises(ValueError):
        compute_mvgc_batch(data, 1, device='cpu')


@pytest.mark.parametrize('device', ['cuda', 'cuda:0', torch.device('cuda:0')])
def test_cuda_fallback(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    x = simulate()
    with pytest.warns(UserWarning, match='falling back'):
        actual = compute_mvgc(x, 2, device=device)
    np.testing.assert_allclose(actual, reference(x, 2, 0.01), atol=1e-10)


def cuda_usable():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    cuda_version = tuple(int(v) for v in torch.version.cuda.split('.')[:2])
    # CUDA 11.8 builds report available on Blackwell but cannot run its kernels.
    return not (torch.cuda.get_device_capability()[0] >= 10 and cuda_version < (12, 8))


@pytest.mark.skipif(not cuda_usable(), reason='CUDA unavailable or build incompatible with GPU')
@pytest.mark.parametrize('alpha', [0.0, 0.01])
def test_cuda_batch_matches_cpu(alpha):
    data = np.stack([simulate(), simulate(seed=43)])
    np.testing.assert_allclose(compute_mvgc_batch(data, 2, alpha, 'cuda:0'),
                               compute_mvgc_batch(data, 2, alpha, 'cpu'), rtol=1e-8, atol=1e-10)


def test_one_channel():
    for engine in [lambda x: compute_mvgc(x, 2, device='cpu'), lambda x: compute_mvgc_cpu(x, 2)]:
        np.testing.assert_array_equal(engine(np.arange(40)[None, :]), np.zeros((1, 1)))


def test_public_and_legacy_exports_match():
    import fast_mvgc
    import pytorch_mvgc
    from pytorch_mvgc.core import compute_mvgc as public_core
    from pytorch_mvgc.cpu_engine import compute_mvgc_cpu as public_cpu
    from pytorch_mvgc.utils import segment_data as public_segment
    assert pytorch_mvgc.__version__ == fast_mvgc.__version__ == '0.1.1'
    assert public_core is fast_mvgc.compute_mvgc is pytorch_mvgc.compute_mvgc
    assert public_cpu is fast_mvgc.compute_mvgc_cpu
    assert public_segment is fast_mvgc.segment_data


@pytest.mark.parametrize('p', [2, 3])
def test_short_data_cpu_torch_consistency(p):
    x = np.random.default_rng(4).normal(size=(3, p+3))
    for alpha in [0, 0.01]:
        np.testing.assert_allclose(compute_mvgc(x, p, alpha, 'cpu'),
                                   reference(x, p, alpha), atol=1e-8)
        np.testing.assert_allclose(compute_mvgc_cpu(x, p, alpha),
                                   reference(x, p, alpha), atol=1e-8)


def test_segments():
    x = np.arange(30).reshape(3, 10)
    actual = segment_data(x.tolist(), 4, 3)
    np.testing.assert_array_equal(actual, np.stack([x[:, :4], x[:, 3:7], x[:, 6:10]]))


@pytest.mark.parametrize('window,step', [(0, 1), (-1, 1), (11, 1), (4, 0), (4, -1), (2.5, 1), (4, True)])
def test_invalid_segmentation(window, step):
    with pytest.raises(ValueError):
        segment_data(np.zeros((3, 10)), window, step)
