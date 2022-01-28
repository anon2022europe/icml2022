from setuptools import setup, find_packages

reqs=[
    "DominantSparseEigenAD",
    "torch",
    "networkx",
    "numpy",
    "scipy",
    "tensorboardX",
    "igraph" # for the leiden
]
packages = find_packages(where=".", exclude=["externals", "*.egg-info", "scripts"])
print(f"Have packages {packages}")
setup(
    name="ggg_utils",
    version="0.0",
    packages=packages,
    url="",
    license="",
    author="anon anons, anon anon, anon anon",
    author_email="",
    description="",
    install_requires=reqs,
)
