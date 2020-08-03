import pyproject
from pyproject import projection_object
import numpy as np
import pandas as pd
import scanpy as sc
import sys


# print("Reading patterns...", file=sys.stderr)
patterns = sc.read_h5ad('patterns_anndata.h5ad')
print(patterns.X.shape, "patterns shape")

# print("Reading target geneset...", file=sys.stderr)
dataset = sc.read_h5ad('test_target.h5ad')
print(dataset.X.shape, "Target shape")
# target.var = target.var.set_index('gene_id')

# print("Doing projection...",file=sys.stderr)
# projection = pyproject.projection.project(target,patterns)

object = projection_object.projection(dataset, patterns, 'CellType', 'gene_id', 12)
object.non_neg_lin_reg(.01, .01)
object.pearsonPlot()
object.UMAP_Projection()
object.featurePlots()

