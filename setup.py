from distutils.core import setup

setup(
    name='pyproject',
    version='0.1.0',
    author='Loyal A. Goff',
    author_email='loyalgoff@jhmi.edu',
    packages=['pyproject', 'pyproject.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/pyproject/',
    license='LICENSE.txt',
    description='Transfer learning framework for single cell gene expression analysis in Python',
    long_description=open('README.txt').read(),
    install_requires=[
        "Django >= 1.1.1",  #Edit accordingly
        "caldav == 0.1.4",  #
    ],
)