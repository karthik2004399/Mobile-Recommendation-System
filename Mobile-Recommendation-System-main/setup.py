from setuptools import setup, find_packages

with open(file='README.md', mode='r', encoding='utf-8') as file:
    long_description = file.read()

__version__ = '0.0.1'

SRC_REPO = 'src'

setup(
    name=SRC_REPO,
    version=__version__,
    description='Movie Recommender System Project',
    packages=find_packages(where=SRC_REPO),
    package_dir={SRC_REPO: SRC_REPO},
    long_description=long_description,
)

