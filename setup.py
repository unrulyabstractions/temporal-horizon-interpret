"""Setup script for temporal-horizon-detection package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="temporal-horizon-detection",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Detecting temporal reasoning scope in LLMs using activation probing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/temporal-horizon-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.30.0",
        "transformer-lens>=1.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "api": [
            "openai>=1.0.0",
            "anthropic>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "thd-generate=scripts.generate_dataset:main",
            "thd-train=scripts.train_probes:main",
            "thd-analyze=scripts.run_circuit_analysis:main",
            "thd-evaluate=scripts.evaluate_model:main",
        ],
    },
)
