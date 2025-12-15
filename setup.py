from setuptools import setup, find_packages

setup(
    name="pytorch-mvgc",                
    version="0.1.0",
    author="nju_sit_hubc",
    author_email="2515094316@qq.com",
    description="A GPU-accelerated Multivariate Granger Causality implementation using PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytorch-mvgc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "torch>=1.9.0",  # 指定最低版本
        "tqdm"
    ],
)