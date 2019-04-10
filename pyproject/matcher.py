#!/usr/bin/env python

##################
# Feature matching/mapping between source (annData) and target (patterns) datasets
##################
import anndata as ad

# class SourceTypeError(AssertionError):
# 	"""Raised if source is not AnnData"""
# 	print("Source data must be a valid AnnData object")

def sourceIsValid(adata):

	# Ensure source data is valid annData object
	try:
		assert isinstance(adata, ad.AnnData)
	except:
		raise SourceTypeError

def getOverlap(adata,patterns=None,idList=None):
	return(adata.var.index.intersection(patterns.var.index))

def filterSource(adata,overlap):
	adata_filtered = adata[:,overlap]
	assert adata_filtered.shape[1] > 0
	return(adata_filtered)

def filterPatterns(patterns,overlap):
	patterns_filtered = patterns[:,overlap]
	assert patterns_filtered.shape[1] > 0
	return(patterns_filtered)
