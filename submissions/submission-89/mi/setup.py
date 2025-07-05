from setuptools import setup, find_namespace_packages

setup(
    name="gmm_v2_mi",
    version="0.1.0",
    packages=find_namespace_packages(include=["gmm_v2.mi*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "gmm_v2_data"
    ],
    author="GMM Team",
    author_email="example@example.com",
    description="GMM mutual information module",
    python_requires=">=3.8",
)
