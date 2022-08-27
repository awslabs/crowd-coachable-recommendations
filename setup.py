from setuptools import find_packages, setup

setup(
    name="ccrec",
    version="1.0",    # will be overwritten by use_scm_version
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "datasets >= 2.4.0",
        "shap >= 0.41.0",
        'psutil',
        'recurrent-intensity-model-experiments @ git+https://github.com/awslabs/recurrent-intensity-model-experiments@main',
    ],
)
