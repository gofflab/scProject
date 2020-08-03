#!/usr/bin/env python

import numpy as np
import sklearn
from anndata import AnnData
from sklearn import linear_model
from sklearn.decomposition import NMF
from . import matcher


class Pattern(AnnData):
    pass


def project(adata, patterns, method="linReg", verbose=True):
    if method == "linReg":
        return project_linreg_(adata, patterns, verbose=verbose)
    else:
        pass


def project_linreg_(adata, patterns, verbose=True):
    overlap = matcher.getOverlap(adata, patterns)
    if verbose:
        print("# of overlapping genes: %d" % (len(overlap)))

    adata_filtered = matcher.filterSource(adata, overlap)
    if verbose:
        print("Filtered Source Dataset:\n" + repr(adata_filtered))

    patterns_filtered = matcher.filterPatterns(patterns, overlap)
    if verbose:
        print("Filtered Pattern Set:\n" + repr(patterns_filtered))

    lm = linear_model.LinearRegression()
    lm.fit(patterns_filtered.X.T, adata_filtered.X.T)
    return lm


def project_NMF(adata, rank):
    model = NMF(n_components=rank, init='random', random_state=0)
    # array = np.absolute(adata.X.T)
    A_Matrix = model.fit_transform(adata.X.T)
    print(A_Matrix.shape)
    return A_Matrix


def non_negative_lin_reg(adata, patterns, alpha, L1, verbose=True):

    color = adata.obs['CellType'].replace({'Astrocyte/Radial Glia': 0,
                                           'Brain Fibroblasts': 1,
                                           'Cycling Neural Progenitor': 2,
                                           'DARPP-32 cells': 3,
                                           'Excitatory Neurons': 4,
                                           'Oligodendrocyte': 5,
                                           'Unknown': 6,
                                           'Vascular Endothelium': 7,
                                           'Vascular Smooth Muscle': 8,
                                           'Interneurons': 9,
                                           'Doublets': 10,
                                           'Microglia': 11})

    overlap = matcher.getOverlap(adata, patterns)
    if verbose:
        print("# of overlapping genes: %d" % (len(overlap)))
    adata_filtered = matcher.filterSource(adata, overlap)
    if verbose:
        print("Filtered Source Dataset:\n" + repr(adata_filtered))

    patterns_filtered = matcher.filterPatterns(patterns, overlap)
    if verbose:
        print("Filtered Pattern Set:\n" + repr(patterns_filtered))
    print(adata_filtered.X.T.shape, "full dataset")
    print(patterns_filtered.X.T.shape, "A-Matrix")

    model = linear_model.ElasticNet(alpha=alpha, l1_ratio=L1, positive=True, max_iter=10000)
    model.fit(patterns_filtered.X.T, adata_filtered.X.T)
    print(model.coef_.shape, "pattern M shape")
    print(model.coef_.dtype)
    col_21 = model.coef_.T[20][:]
    print(col_21.shape)
    ints = col_21.astype('int64')

    umap_obj = umap.UMAP(n_neighbors=75)
    nd = umap_obj.fit_transform(model.coef_)
    print(nd.shape, "umap shape")

    plt.scatter(nd[:, 0], nd[:, 1],
                c=[sns.color_palette("Paired", n_colors=12)[x] for x in color], s=5)
    plt.title("UMAP projection of pattern matrix", fontsize=24)
    plt.show()
    return model
