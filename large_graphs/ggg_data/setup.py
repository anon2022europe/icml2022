from setuptools import setup, find_packages

reqs=[
    "torch",  # cluster compatible versions
    # "tensorflow-cpu",
    "matplotlib",
    "numpy",
    "tensorboardX",
    "scipy",
    "networkx",
    # "rdkit" => conda
    # datashader => conda
    "pandas",
    "ogb",
    "pygraphviz",
    "pydot",
    "ggg_utils"

]
packages = find_packages(where=".", exclude=["externals", "*.egg-info", "scripts"])
print(f"Have packages {packages}")
setup(
    name="ggg_data",
    version="0.0",
    packages=packages,
    url="",
    license="",
    author="anon anons, anon anon, anon anon",
    author_email="",
    description="",
    install_requires=reqs,
)
