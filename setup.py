from setuptools import setup, find_namespace_packages


def read_requirements(path: str = "requirements.txt") -> list[str]:
    """requirements.txt without comments and blank lines."""
    with open(path) as f:
        lines = (line.split("#", 1)[0].strip() for line in f)
        return [line for line in lines if line]


setup(
    name="air_hockey_wm",
    version="0.1.0",
    # wm/ is the library; tools/ holds the pipeline entry points (python -m tools....) and the
    # helpers the notebook imports. tools/ has no __init__.py, hence namespace packages.
    packages=find_namespace_packages(include=["wm", "wm.*", "tools", "tools.*"]),
    url="https://github.com/RobvanGastel/multiplayer-world-model",
    license="",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black",
        ]
    },
    author="Rob van Gastel",
    description="A small-scale multiplayer interactive world model (MIRA) for two-player air hockey.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
