from setuptools import setup
import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    packages=setuptools.find_packages(),
    install_requires=requirements,
    name='scProject',
    version='1.1.3',
    author='Asher Baraban, Charles Shin, Genevieve Stein-O\'Brien, Loyal Goff',
    author_email='asher.baraban@wustl.edu, cshin12@jhu.edu',
    scripts=[],
    url="https://github.com/gofflab/scProject",
    license='MIT',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable'
    ]
)
