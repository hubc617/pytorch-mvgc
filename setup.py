from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="pytorch-mvgc",
    version="0.1.1",
    author="nju_sit_hubc",
    author_email="2515094316@qq.com",
    description="A GPU-accelerated Multivariate Granger Causality implementation using PyTorch.",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/hubc617/pytorch-mvgc",
    packages=find_packages(include=["fast_mvgc", "fast_mvgc.*", "pytorch_mvgc", "pytorch_mvgc.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=["numpy>=1.20.0", "torch>=1.9.0", "tqdm>=4.60.0"],
    extras_require={"test": ["pytest>=7"]},
)
