from setuptools import find_packages, setup
import subprocess, re


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        match = re.search(r"Build cuda_(\d+).(\d+)", output)
        return match.group(1), match.group(2)
    except FileNotFoundError:
        return None


setup(
    name="ccrec",
    version="1.0",  # will be overwritten by use_scm_version
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
