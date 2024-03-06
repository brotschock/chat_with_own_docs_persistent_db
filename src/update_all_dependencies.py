import os
import subprocess

# Get the list of all installed packages
installed_packages = subprocess.check_output(["pip", "list", "--format=freeze"]).decode("utf-8").split("\n")

# Remove the version number from each package name
packages = [package.split("==")[0] for package in installed_packages if "==" in package]

# Upgrade each package
for package in packages:
    os.system(f"pip install --upgrade {package}")