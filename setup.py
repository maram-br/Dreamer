from setuptools import setup, find_packages

setup(
    name="sdbs",
    version="0.1.0",
    description="S-DBS Dreamer-PPO: Safe Diverse Beam Search for Autonomous Driving",
    packages=find_packages(include=["sdbs", "sdbs.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pygame",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev":   ["torch>=2.0", "pygame"],
    },
)
