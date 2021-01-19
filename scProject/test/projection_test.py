from scProject import matcher
from scProject import viz
from scProject import stats
from scProject import rg
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

#matcher.orthologMapper()
# print("Reading patterns...", file=sys.stderr)
patterns = sc.read_h5ad('patterns_anndata.h5ad')
print(patterns.X.shape, "patterns shape")

# print("Reading target geneset...", file=sys.stderr)
dataset = sc.read_h5ad('test_target.h5ad')
print(dataset.X.shape, "Target shape")
# target.var = target.var.set_index('gene_id')

# print("Doing projection...",file=sys.stderr)
# projection = scProject.projection.project(target,patterns)


dataset_filtered, patterns_filtered = matcher.filterAnnDatas(dataset, patterns, 'gene_id')
#dataset_filtered, patterns_filtered = matcher.logTransform(dataset_filtered)

# stats.geneSelectivity(patterns_filtered, 'ENSMUSG00000036887', 24)
# projection_object.geneSelectivity(patterns_filtered, 'ENSMUSG00000036905', 5)
# projection_object.geneSelectivity(patterns_filtered, 'ENSMUSG00000036896', 6)
rg.NNLR_ElasticNet(dataset_filtered, patterns_filtered, 'retinaProject', .01, .001)
# stats.geneDriver(dataset_filtered, patterns_filtered, 'ENSMUSG00000036896', "CellType", "Brain Fibroblasts", "retinaProject")
microglia  = dataset_filtered[dataset_filtered.obs['CellType'].isin(['Microglia'])]
print(microglia.shape)
C1 = microglia[microglia.obsm['retinaProject'][:, 4] > 1]
C2 = microglia[microglia.obsm['retinaProject'][:, 4] <1]
print(C1.shape)
print(C2.shape)
stats.BonferroniCorrectedDifferenceMeans(C1, C2, .9, 'gene_short_name')



# projection_object.NNLR_positive_Lasso(dataset_filtered, patterns_filtered, 'retinaProject', 0.005)
# projection_object.NNLR_LeastSquares(dataset_filtered, patterns_filtered, 'retinaProject')
# viz.pearsonMatrix(dataset_filtered, patterns_filtered, 'CellType', 12, 'retinaProject', 'PearsonRetina',
#                 True, row_cluster=True)
print("pearson")
#viz.UMAP_Projection(dataset_filtered, 'CellType', 'retinaProject', 'retinaUMAP', 10)
# stats.featureImportance(dataset_filtered, 80, 'retinaProject')
#viz.featurePlots(dataset_filtered, [9], 'retinaProject', 'retinaUMAP')
