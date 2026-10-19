# Publishing a release

The `Publish to PyPI` workflow builds and checks distributions before uploading
them through PyPI Trusted Publishing. It uses short-lived OIDC credentials;
no PyPI password or long-lived API token is stored in GitHub.

Configure this publisher for the existing `pytorch-mvgc` project on PyPI:

| Setting | Value |
| --- | --- |
| Owner | `hubc617` |
| Repository | `pytorch-mvgc` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

The GitHub `pypi` environment allows deployments from `main` only. The workflow
also refuses to build on other refs and only grants `id-token: write` to the
upload job. The upload job consumes the checked build artifacts.

To release:

1. Update the version in `setup.py` and `fast_mvgc/__init__.py`, and the version
   assertion in the regression tests. Review and merge the changes into `main`.
2. Run the complete regression suite on a compatible CUDA machine. GPU parity
   cases are skipped on CPU-only GitHub runners; local CUDA validation is still
   required for changes to GPU computation.
3. Open GitHub Actions, select **Publish to PyPI**, choose **Run workflow** on
   `main`, and enter the declared version. A version mismatch aborts the build.
4. Confirm both wheel and sdist appear on PyPI and install the published wheel
   in a compatible environment to check imports and computation.

Version 0.1.1 was validated on an RTX 5070 Laptop GPU (sm_120), Python 3.12.13,
PyTorch 2.11.0+cu128 and NumPy 2.5.2: all 64 tests passed, including both CUDA
parity cases. The 128-channel, 10-window GPU example also produced finite,
nonnegative matrices with zero diagonals.

See [PyPI's Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
