from setuptools import setup

setup(
    name='ProjectPy',
    version='0.1.0',
    author='Loyal A. Goff',
    author_email='loyalgoff@jhmi.edu',
    packages=['ProjectPy', 'pyproject.test'],
    scripts=[],
    url='http://github.com/gofflab/pyproject/',
    license='MIT',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.md').read(),
    install_requires=[
        "scanpy >= 1.4",  #Edit accordingly
        "anndata >= 0.6", #
    ],
)
