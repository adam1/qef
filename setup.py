from setuptools import setup, find_packages

setup(
    name="qef",
    version="0.1.0",
    description="Quantum Error Forms - Computing Hermitian forms for quantum error correction",
    author="Adam",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "sympy>=1.12",
        "h5py>=3.8.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
