#!/usr/bin/env python
from . import matcher
from scipy import stats
import numpy as np


def pearsonPlot(adata, patterns, cellTypeColumnName):
    ints = matcher.mapCellNamesToInts(adata, cellTypeColumnName)
    matrix = np.zeros([range(adata.obs[cellTypeColumnName].unique()), ints.shape[0]])
    for i in range(ints.shape[0]):
        val = ints[i]
        matrix[val][i] = 1


