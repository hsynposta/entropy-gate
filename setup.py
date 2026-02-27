from setuptools import setup, find_packages

setup(
    name="entropy-gate",
    version="0.1.0",
    author="Hiram Aydin",
    description="Hybrid Geometric-Entropy Gating for OOD-robust neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy>=1.20",
    ],
    extras_require={
        "demo": ["matplotlib>=3.4"],
        "dev": ["pytest", "matplotlib"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
