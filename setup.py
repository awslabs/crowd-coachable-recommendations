from setuptools import find_packages, setup

setup(
    name="ccrec",
    version="1.0",    # will be overwritten by use_scm_version
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        # please install the latest version of rime.
        "datasets >= 2.4.0",
    ],
)
