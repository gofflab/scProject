from scProject import projection_object
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
# projection = scProject.projection.project(target,patterns)


dataset_filtered, patterns_filtered = projection_object.filterAnnDatas(dataset, patterns, 'gene_id')
projection_object.NNLR_ElasticNet(dataset_filtered, patterns_filtered, 'retinaProject', alpha=.01, L1=.01)
projection_object.pearsonMatrix(dataset_filtered, patterns_filtered, 'CellType', 12, 'retinaProject', 'PearsonRetina', True)
projection_object.UMAP_Projection(dataset_filtered, 'CellType', 'retinaProject', 'retinaUMAP', 12)
projection_object.featurePlots(dataset_filtered, 80, 'retinaProject', 'retinaUMAP')