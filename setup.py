"""Setup script for glmocr package.

NOTE: This file is deprecated. Use pyproject.toml instead for package installation.
This file is kept for backward compatibility only.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Get README
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# Get requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="glmocr",
    version="0.1.1",
    author="ZHIPUAI",
    author_email="info@zhipuai.cn",
    description="GLM-OCR - Optical Character Recognition powered by GLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZHIPUAI/glm-ocr",
    # Packages
    packages=find_packages(exclude=["tests", "tests.*", "demo", "demo.*", "docs"]),
    # Python version requirement
    python_requires=">=3.8",
    # Dependencies
    install_requires=requirements,
    # Include non-Python files
    include_package_data=True,
    package_data={
        "glmocr": [
            "config.yaml",
        ],
    },
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # CLI entrypoints
    entry_points={
        "console_scripts": [
            "glmocr=glmocr.cli:main",
        ],
    },
)
