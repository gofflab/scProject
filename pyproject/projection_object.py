from . import matcher
from sklearn import linear_model
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import anndata as ad
import scanpy as sc


# Super Class
class projection:
    def __init__(self, dataset, patterns, cellTypeColumnName, genecolumnname, num_cell_types):
        self.dataset = dataset
        self.patterns = patterns
        dataset.var = dataset.var.set_index(genecolumnname)
        overlap = dataset.var.index.intersection(patterns.var.index)
        self.dataset_filtered = dataset[:, overlap]
        print(self.dataset_filtered.shape, "dataset filter shape")
        self.patterns_filtered = patterns[:, overlap]
        print(self.patterns_filtered.shape, "patterns filter shape")
        self.cellTypeColumnName = cellTypeColumnName
        self.model = None
        self.pearsonMatrix = None
        self.num_cell_types = num_cell_types
        self.num_patterns = patterns.X.shape[0]
        self.UMAP_COORD = None

    def non_neg_lin_reg(self, alpha, L1, iterations=10000):
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=L1, max_iter=iterations)
        model.fit(self.patterns_filtered.X.T, self.dataset_filtered.X.T)
        self.model = model

    def pearsonPlot(self, plot=True):
        color = matcher.mapCellNamesToInts(self.dataset_filtered, self.cellTypeColumnName)
        matrix = np.zeros([self.num_cell_types, color.shape[0]])
        pearson_matrix = np.empty([self.patterns_filtered.X.shape[0], self.num_cell_types])
        for i in range(color.shape[0]):
            cell_type = color[i]
            matrix[cell_type][i] = 1
        for i in range(self.patterns_filtered.X.shape[0]):
            pattern = np.transpose(self.model.coef_)[:][i]
            for j in range(color.unique().shape[0]):
                cell_type = matrix[j]
                correlation = stats.pearsonr(pattern, cell_type)
                pearson_matrix[i][j] = correlation[0]
        self.pearsonMatrix = pearson_matrix
        if plot:
            plt.title("Pearson Plot", fontsize=24)
            graphic = sns.heatmap(pearson_matrix)
            plt.show()

    def UMAP_Projection(self, n_neighbors=10, metric='euclidean', plot=True, color="Paired"):
        color = matcher.mapCellNamesToInts(self.dataset_filtered, self.cellTypeColumnName)
        umap_obj = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
        nd = umap_obj.fit_transform(self.model.coef_)
        if plot:
            plt.scatter(nd[:, 0], nd[:, 1],
                        c=[sns.color_palette("Paired", n_colors=12)[x] for x in color], s=.5)
            plt.title("UMAP Projection of Pattern Matrix", fontsize=24)
            plt.show()
        self.UMAP_COORD = nd

    def featurePlots(self):
        for i in range(self.num_patterns):
            feature = self.model.coef_[:, i]
            plt.title("Feature " + str(i + 1), fontsize=24)
            plt.scatter(self.UMAP_COORD[:, 0], self.UMAP_COORD[:, 1], c=feature, cmap='jet', s=.5)
            plt.colorbar()
            print(np.count_nonzero(feature))
            plt.show()


# Here I refactored the code to take in AnnData Objects
def filterAnnDatas(dataset, patterns, geneColumnName):
    dataset.var = dataset.var.set_index(geneColumnName)
    overlap = dataset.var.index.intersection(patterns.var.index)
    dataset_filtered = dataset[:, overlap]
    print(dataset_filtered.shape, "dataset filter shape")
    patterns_filtered = patterns[:, overlap]
    print(patterns_filtered.shape, "patterns filter shape")
    return dataset_filtered, patterns_filtered


def non_neg_lin_reg(dataset_filtered, patterns_filtered, projectionName, alpha, L1, iterations=10000):
    model = linear_model.ElasticNet(alpha=alpha, l1_ratio=L1, max_iter=iterations)
    model.fit(patterns_filtered.X.T, dataset_filtered.X.T)
    dataset_filtered.obsm[projectionName] = model.coef_


def pearsonMatrix(dataset_filtered, patterns_filtered, cellTypeColumnName, num_cell_types, projectionName, plotName,
                  plot):
    color = matcher.mapCellNamesToInts(dataset_filtered, cellTypeColumnName)
    matrix = np.zeros([num_cell_types, color.shape[0]])
    pearson_matrix = np.empty([patterns_filtered.X.shape[0], num_cell_types])
    for i in range(color.shape[0]):
        cell_type = color[i]
        matrix[cell_type][i] = 1
    for i in range(patterns_filtered.X.shape[0]):
        pattern = np.transpose(dataset_filtered.obsm[projectionName])[:][i]
        for j in range(color.unique().shape[0]):
            cell_type = matrix[j]
            correlation = stats.pearsonr(pattern, cell_type)
            pearson_matrix[i][j] = correlation[0]
            dataset_filtered.uns[plotName] = pearson_matrix
    # dataset_filtered.obsm['Pearson'] = pearson_matrix
    if plot:
        pearsonViz(dataset_filtered, plotName)


def pearsonViz(dataset_filtered, plotName):
    plt.title("Pearson Plot", fontsize=24)
    graphic = sns.heatmap(dataset_filtered.uns[plotName])
    plt.show()


def UMAP_Projection(dataset_filtered, cellTypeColumnName, projectionName, UMAPName, n_neighbors, metric='euclidean',
                    plot=True, color='Paired'):
    color = matcher.mapCellNamesToInts(dataset_filtered, cellTypeColumnName)
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
    nd = umap_obj.fit_transform(dataset_filtered.obsm[projectionName])
    if plot:
        plt.scatter(nd[:, 0], nd[:, 1],
                    c=[sns.color_palette("Paired", n_colors=12)[x] for x in color], s=.5)
        plt.title("UMAP Projection of Pattern Matrix", fontsize=24)
        plt.show()
    dataset_filtered.obsm[UMAPName] = nd


def featurePlots(dataset_filtered, num_patterns, projectionName, UMAPName):
    for i in range(num_patterns):
        pattern_matrix = dataset_filtered.obsm[projectionName]
        feature = pattern_matrix[:, i]
        plt.title("Feature " + str(i + 1), fontsize=24)
        plt.scatter(dataset_filtered.obsm[UMAPName][:, 0], dataset_filtered.obsm[UMAPName][:, 1], c=feature, cmap='jet',
                    s=.5)
        plt.colorbar()
        print(np.count_nonzero(feature))
        plt.show()


def saveProjections(dataset_filtered, datasetFileName):
    dataset_filtered.write.h5ad(datasetFileName)
