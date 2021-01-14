from setuptools import setup
import setuptools

setup(
    packages=setuptools.find_packages(),
    name='scProject',
    version='1.0.9.7',
    author='Asher Baraban, Genevieve Stein-O\'Brien, Loyal Goff',
    author_email='loyalgoff@jhmi.edu',
    scripts=[],
    url="https://github.com/gofflab/ProjectPy.git",
    license='MIT',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta'
    ]
)
