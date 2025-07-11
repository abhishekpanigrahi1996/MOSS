from setuptools import setup, find_namespace_packages

setup(
    name="gmm_v2_data",
    version="0.1.0",
    packages=find_namespace_packages(include=["gmm_v2.data*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch"
    ],
    author="GMM Team",
    author_email="example@example.com",
    description="GMM data generation module",
    python_requires=">=3.8",
)
