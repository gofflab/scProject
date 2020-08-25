from setuptools import setup
import setuptools

setup(
    packages=setuptools.find_packages(),
    name='scProject',
    version='1.0.0',
    author='Asher Baraban, Loyal Goff',
    author_email='loyalgoff@jhmi.edu',
    scripts=[],
    url="https://github.com/gofflab/ProjectPy.git",
    license='MIT',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.md').read(),
    classifiers=[
        'Development Status :: 3 - Alpha'
    ]
)
