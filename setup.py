from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="synthclinic",
    version="0.1.0",
    description=(
        "SynthClinic: A modular generative AI framework for realistic synthetic "
        "medical data — ECG signals, MRI images, clinical notes, and tabular lab results."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SynthClinic Research",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "experiments*"]),
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4", "black>=23.0", "ruff>=0.1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
