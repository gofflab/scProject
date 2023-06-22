from setuptools import setup
import setuptools

setup(
    packages=setuptools.find_packages(),
    name='scProject',
    version='1.0.9.8996',
    author='Asher Baraban, Genevieve Stein-O\'Brien, Loyal Goff',
    author_email='asher.baraban@wustl.edu',
    scripts=[],
    url="https://github.com/gofflab/scProject",
    license='MIT',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta'
    ]
)
