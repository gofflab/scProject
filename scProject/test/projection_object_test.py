import scProject
import scanpy as sc
patterns = sc.read_h5ad('patterns_anndata.h5ad')
target = sc.read_h5ad('test_target.h5ad')
adata.var = adata.var.set_index('gene_id')
