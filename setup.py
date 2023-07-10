from setuptools import find_packages, setup
import subprocess, re


setup(
    name="ccrec",
    version="1.0",
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "datasets >= 2.4.0",
        "shap >= 0.41.0",
        "pytest",
        "flaky",
        "tensorboard",
        "pytorch-lightning < 2.0.0",
    ],
)
