from setuptools import setup, find_packages

reqs = [
    "torch",  # cluster compatible versions
    # "tensorflow-cpu",
    "sacred",
    "matplotlib",
    "numpy",
    "sparsemax",
    "tensorboardX",
    "scipy",
    "networkx",
    "ipython",
    "ipdb",
    "pdbpp",
    "jupyter",
    "pyemd",
    # "rdkit" => conda
    # datashader => conda
    "umap-learn[plot]",
    "pandas",
    "colorcet",
    "DominantSparseEigenAD",
    "pytorch_lightning<0.9",
    "imageio",
    "seaborn",
    "ogb",
]
# uncomment if running in modern cluster
with open("requirements.txt", "r") as f:
    reqs = f.readlines()
packages = find_packages(where=".", exclude=["externals", "*.egg-info", "scripts"])
print(f"Have pack,ages {packages}")
setup(
    name="ggg",
    version="0.0",
    packages=packages,
    url="",
    license="",
    author="anon anons, anon anon, anon anon",
    author_email="",
    description="",
    install_requires=reqs,
)
