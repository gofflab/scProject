#!/usr/bin/env python

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn import linear_model
from . import matcher


class Pattern(AnnData):
	pass


def project(adata,patterns,method="linReg",verbose=True):
	if(method=="linReg"):
		return(project_linreg_(adata,patterns,verbose=verbose))
	else:
		pass

def project_linreg_(adata,patterns,verbose=True):
	overlap = matcher.getOverlap(adata,patterns)
	if verbose: print("# of overlapping genes: %d" % (len(overlap)))
	
	adata_filtered = matcher.filterSource(adata,overlap)
	if verbose: print("Filtered Source Dataset:\n" + repr(adata_filtered))
	
	patterns_filtered = matcher.filterPatterns(patterns,overlap)
	if verbose: print("Filtered Pattern Set:\n" + repr(patterns_filtered))
	
	lm = linear_model.LinearRegression()
	lm.fit(patterns_filtered.X.T,adata_filtered.X.T)
	return(lm)

