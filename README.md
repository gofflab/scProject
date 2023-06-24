# scProject # 

## Overview ##

Transfer learning for gene expression signatures(Python implementation of ProjectR).  

scProject is a software package that allows biologists to use transfer learning to understand the usage of previously discovered features in genomic datasets. Tools like NNMF allow biologists to find a set of vectors that linearly combine to approximately reconstruct a dataset. These features can have biological meaning as a pathway, cell type, gene program, cell feature, etc. scProject leverages scikit-learn’s and scipy’s regression tools to effectively perform a change of basis with the new basis being the features discovered from NNMF or another previous analysis tool. For example, the values in the first column of the pattern matrix are the coefficients of the features to approximate the first sample in the dataset. This analysis allows the user to see the usage of features from a previously examined dataset and see if those same features are utilized in a new dataset. Using UMAP-learn to decrease the dimensionality of the regressed pattern matrix allows the user to visualize the pattern matrix in intuitive and informative ways.  

## Contents ##

Name          | Link
------------- | -------------
scProject API | https://scproject.readthedocs.io/en/master/contents.html
Regression    | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.rg
Visualization | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.viz
Statistics    | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.stats
Utilities     | https://scproject.readthedocs.io/en/master/contents.html#module-scProject.matcher
Tutorial      | https://scproject.readthedocs.io/en/master/tutorial.html

## Indices and tables ##

Name          | Link
------------- | -------------
Module Index  | https://scproject.readthedocs.io/en/master/py-modindex.html
Search Page   | https://scproject.readthedocs.io/en/master/search.html

## Additional Information ##

For more instructions and guidelines, read the [documentation](https://scproject.readthedocs.io/en/master/).
