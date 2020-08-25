#!/usr/bin/env python

##################
# Feature matching/mapping between source (annData) and target (patterns) datasets
##################
import anndata as ad


# class SourceTypeError(AssertionError):
# 	"""Raised if source is not AnnData"""
# 	print("Source data must be a valid AnnData object")

def sourceIsValid(adata):
    """Checks whether adata is an AnnData object

    :param adata: AnnData object
    :return: SourceTypeError if adata is not an instance of an AnnData
    """
    # Ensure source data is valid annData object
    try:
        assert isinstance(adata, ad.AnnData)
    except:
        raise SourceTypeError

# these are not currently used in scProject
def getOverlap(adata, patterns=None, idList=None):
    return adata.var.index.intersection(patterns.var.index)


def filterSource(adata, overlap):
    adata_filtered = adata[:, overlap]
    assert adata_filtered.shape[1] > 0
    return adata_filtered


def filterPatterns(patterns, overlap):
    patterns_filtered = patterns[:, overlap]
    assert patterns_filtered.shape[1] > 0
    return patterns_filtered


# Parameters
# adata : AnnData object
# cellTypeColumnName : index of the cell types in .obs
# Return-void
# Purpose of the Method:
# This method performs an elastic net regression from Sklearn. The "discovered" pattern matrix is stored in
# the dataset_filtered.obsm under the paramater projectionName
def mapCellNamesToInts(adata, cellTypeColumnName):
    """Maps each cell type to an integer. This is used as a helper for coloring plots

    :param adata: AnnData object
    :param cellTypeColumnName: index of where cell type is stored in adata.obs
    :return:
    """
    print(adata.obs[cellTypeColumnName].unique())
    zipper = zip(adata.obs[cellTypeColumnName].unique(), range(adata.obs[cellTypeColumnName].unique().shape[0]))
    dictionary = dict(zipper)
    new_obs = adata.obs[cellTypeColumnName].replace(dictionary)
    return new_obs
