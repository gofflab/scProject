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
def getOverlap(dataset, patterns):
    """ Convenience function for overlap of genes

    :param dataset: Anndata object cells x genes
    :param patterns: Anndata object features x genes
    :return: Overlap of genes
    """
    return dataset.var.index.intersection(patterns.var.index)


def filterSource(dataset, overlap):
    """ Convenience function for using an inputted set of genes

    :param dataset: Anndata object cells x genes
    :param overlap: list-like of genes
    :return: Filtered dataset (AnnData)
    """
    dataset_filtered = dataset[:, overlap]
    assert dataset_filtered.shape[1] > 0
    return dataset_filtered


def filterPatterns(patterns, overlap):
    """ Convenience function for using an inputted set of genes

    :param patterns: Anndata object features x genes
    :param overlap: list-like of genes
    :return: Filtered patterns (AnnData)
    """
    patterns_filtered = patterns[:, overlap]
    assert patterns_filtered.shape[1] > 0
    return patterns_filtered


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


def filterAnnDatas(dataset, patterns, geneColumnName):
    """ This method filters the patterns and the dataset to only include overlapping genes

    :param dataset: Anndata object cells x genes
    :type dataset: AnnData object
    :param patterns: Anndata object features x genes
    :param geneColumnName: index for where the gene names are kept in .var
    :return: A tuple of two filtered AnnData objects
    """

    sourceIsValid(dataset)  # Make sure dataset is an AnnData object
    sourceIsValid(patterns)  # Make sure patterns is an AnnData object
    dataset.var = dataset.var.set_index(geneColumnName)
    overlap = dataset.var.index.intersection(patterns.var.index)
    dataset_filtered = dataset[:, overlap]
    print(dataset_filtered.shape, "dataset filter shape")
    patterns_filtered = patterns[:, overlap]
    print(patterns_filtered.shape, "patterns filter shape")
    return dataset_filtered, patterns_filtered
