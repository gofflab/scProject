#!/usr/bin/env python

##################
# Feature matching/mapping between source (annData) and target (patterns) datasets
##################
import anndata as ad
import pandas as pd
import numpy as np


# class SourceTypeError(AssertionError):
# 	"""Raised if source is not AnnData"""
# 	print("Source data must be a valid AnnData object")

def sourceIsValid(adata):
    # Ensure source data is valid annData object
    try:
        assert isinstance(adata, ad.AnnData)
    except:
        raise SourceTypeError


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


def mapCellNamesToInts(adata, cellTypeColumnName):
    print(adata.obs[cellTypeColumnName].unique())
    zipper = zip(adata.obs[cellTypeColumnName].unique(), range(adata.obs[cellTypeColumnName].unique().shape[0]))
    dictionary = dict(zipper)
    new_obs = adata.obs[cellTypeColumnName].replace(dictionary)
    return new_obs
