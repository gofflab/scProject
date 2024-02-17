<img src="https://raw.githubusercontent.com/gofflab/scProject/master/docs/scProject-logo.jpg" >

<img alt="PyPI" src="https://img.shields.io/pypi/v/scProject"> [![Downloads](https://static.pepy.tech/personalized-badge/scproject?period=total&units=international_system&left_color=black&right_color=lightgrey&left_text=Downloads)](https://pepy.tech/project/scproject) <img alt="PyPI - License" src="https://img.shields.io/pypi/l/scProject?color=%237003FF"> <img alt="Read the Docs" src="https://img.shields.io/readthedocs/scproject?color=%230b5394"> 

# scProject # 

## Overview ##

Transfer learning for gene expression signatures in Python.

## Description ##

scProject is a software package that allows biologists to use transfer learning to understand the usage of previously discovered features in genomic datasets. Tools like NNMF allow biologists to find a set of vectors that linearly combine to approximately reconstruct a dataset. These features can have biological meaning as a pathway, cell type, gene program, cell feature, etc.  

Built on top of `ProjectR`, scProject leverages `scikit-learn`’s and `scipy`’s regression tools to effectively perform a change of basis with the new basis being the features discovered from NNMF or another previous analysis tool. For example, the values in the first column of the pattern matrix are the coefficients of the features to approximate the first sample in the dataset. This analysis allows the user to see the usage of features from a previously examined dataset and see if those same features are utilized in a new dataset. Using `umap-learn` to decrease the dimensionality of the regressed pattern matrix allows the user to visualize the pattern matrix in intuitive and informative ways.  

## Installation ##

scProject package can be installed by the following code:  

```python
pip install scProject
```

It is recommended to install the package and run the code in a virtual environment (venv). Use the code below to create and activate venv:  

```python
python -m venv env
source env/bin/activate
```

## Contents ##

Name          | Link
------------- | -------------
scProject API | https://scproject.readthedocs.io/en/master/contents.html
Regression    | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.rg
Visualization | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.viz
Statistics    | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.stats
Utilities     | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.matcher

## FAQ

<details>
  <summary>
    I'm getting NumbaDeprecationWarning while importing scProject in Python. What should I do?
  </summary>  
  <br />
  
  ```python
  NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' 
decorator. The implicit default value for this argument is currently False, but it will be changed to True in 
Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-
object-mode-fall-back-behaviour-when-using-jit for details.
    @numba.jit()
  ```

  The following error is coming from `umap` package, not `scProject`. Please ignore this warning message.
</details>

## Additional Information ##

For more instructions and guidelines, read the [documentation](https://scproject.readthedocs.io/en/master/).  
If you would like to leave feedback, please contact [cshin12@jhu.edu](mailto:cshin12@jhu.edu) and [gsteinobrien@jhmi.edu](mailto:gsteinobrien@jhmi.edu).

Copyright © 2022-2024 Goff Lab, Stein-O'Brien Lab. All rights reserved.
