from setuptools import setup

setup(
    name='ProjectPy',
    version='0.1.3',
    author='Asher Baraban, Loyal Goff',
    author_email='loyalgoff@jhmi.edu',
    packages=['ProjectPy', 'ProjectPy.test'],
    scripts=[],
    url="https://github.com/gofflab/ProjectPy.git",
    license='MIT',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.md').read(),
    install_requires=[
        "anndata >= 0.7",  #
        "numpy >= 1.18.1",  #
        "pandas >= 1.0.1",
        "scipy >= 1.4.1",
        "seaborn >= 0.10.0",  #
        "matplotlib >= 3.1.3",
        "umap-learn >=0.4.0",
        "scikit-learn>=0.21.2",
    ],
)
