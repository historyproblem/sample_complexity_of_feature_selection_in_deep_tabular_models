from setuptools import setup, find_packages

setup(
    name="net_complexity",
    version="0.1.0",
    author="...",
    description="...",
    # long_description=open("README.rst").read(),
    long_description_content_type="...",
    url="...",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch==2.1.2",
        "torchvision==0.16.2",
        "hydra-core",
        "omegaconf",
        "tqdm",
        "mlflow",
        "numpy",
        "pandas",
        "optuna",
    ],
)
