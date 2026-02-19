from setuptools import setup, find_packages

subpkgs = find_packages(where=".")  # finds: detection, star, flares, ...
packages = ["allesfast"] + ["allesfast." + p for p in subpkgs]
package_dir = {"allesfast": "."}
for p in subpkgs:
    package_dir["allesfast." + p] = p

setup(
    packages=packages,
    package_dir=package_dir,
)
