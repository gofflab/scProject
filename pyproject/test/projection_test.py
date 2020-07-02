import pyproject
import numpy as np
import pandas as pd
import scanpy as sc
import sys


print("Reading patterns...",file=sys.stderr)
patterns = sc.read_h5ad('patterns_anndata.h5ad')

print("Reading target geneset...",file=sys.stderr)
target = sc.read_h5ad('test_target.h5ad')
target.var = target.var.set_index('gene_id')

print("Doing projection...",file=sys.stderr)
projection = pyproject.projection.project(target,patterns)

project_positive = non_negative_lin_reg(target, patterns, .1, .1, verbose=False)
