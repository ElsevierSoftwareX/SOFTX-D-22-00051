"""
Setup script for ORNL SANS reduction
"""
# from __future__ import absolute_import, division, print_function
import os
from setuptools import setup, find_packages
import versioneer

THIS_DIR = os.path.dirname(__file__)

# get list of scripts to install
scripts = [
    os.path.join(root, f)
    for root, _, files in os.walk(os.path.join(THIS_DIR, "scripts"))
    for f in files
    if f.endswith(".py")
]


def read_requirements_from_file(filepath):
    """Read a list of requirements from the given file and split into a
    list of strings. It is assumed that the file is a flat
    list with one requirement per line.
    :param filepath: Path to the file to read
    :return: A list of strings containing the requirements
    """
    with open(filepath, "r") as req_file:
        return req_file.readlines()


install_requires = read_requirements_from_file(
    os.path.join(THIS_DIR, "requirements.txt")
)
test_requires = read_requirements_from_file(
    os.path.join(THIS_DIR, "requirements_dev.txt")
)

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Natural Language :: English",
    "Operating System :: Linux",
    "Programming Language :: Python",
]

setup(
    name="drtsans",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Data Reduction Toolkit SANS reduction",
    url="https://http://www.mantidproject.org",
    long_description="""ORNL SANS reduction""",
    license="Apache License 2.0",
    zip_safe=False,
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["tests", "tests.*"]),
    scripts=scripts,
    package_dir={},
    package_data={
        "drtsans": [
            "configuration/schema/*.json",
            "mono/biosans/cg3_to_nexus_mapping.yml",
        ]
    },
    install_requires=install_requires,
)
