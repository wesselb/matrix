from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "plum-dispatch>=0.2.3",
    "backends>=0.5.0",
    "wbml>=0.1.3",
    "algebra",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
