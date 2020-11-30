from setuptools import find_packages, setup

requirements = ["numpy>=1.16", "backends", "plum-dispatch", "wbml>=0.1.3", "algebra"]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
