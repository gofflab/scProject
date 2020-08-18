import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from scipy import stats
from scipy.optimize import nnls
from sklearn import linear_model
from . import matcher


# Super Class- Not in use all refactored below
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


# Parameters
# dataset : Anndata object cells x genes
# patterns : Anndata object features x genes
# Return
# Returns a tuple of the two filtered AnnData objects
# Purpose of the Method:
# The point of this method is to only use the overlapping genes of the two anndata objects
def filterAnnDatas(dataset, patterns, geneColumnName):
    matcher.sourceIsValid(dataset)  # Make sure dataset is an AnnData object
    matcher.sourceIsValid(patterns)  # Make sure patterns is an AnnData object
    dataset.var = dataset.var.set_index(geneColumnName)
    overlap = dataset.var.index.intersection(patterns.var.index)
    dataset_filtered = dataset[:, overlap]
    print(dataset_filtered.shape, "dataset filter shape")
    patterns_filtered = patterns[:, overlap]
    print(patterns_filtered.shape, "patterns filter shape")
    return dataset_filtered, patterns_filtered


# Parameters
# dataset_filtered : Anndata object cells x genes
# patterns_filtered : Anndata object features x genes
# projectionName : String which is the index of the projection in obsm
# alpha : regularization parameter
# L1 : regularization parameter
# iterations : number of iterations while doing the regression
# Return-void
# Purpose of the Method:
# This method performs an elastic net regression from Sklearn. The "discovered" pattern matrix is stored in
# the dataset_filtered.obsm under the paramater projectionName
def NNLR_ElasticNet(dataset_filtered, patterns_filtered, projectionName, alpha, L1, iterations=10000):
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
    model = linear_model.ElasticNet(alpha=alpha, l1_ratio=L1, max_iter=iterations, positive=True)
    model.fit(patterns_filtered.X.T, dataset_filtered.X.T)
    dataset_filtered.obsm[projectionName] = model.coef_
    print(model.coef_.shape)


# still experimenting with this but same idea as NNLR_Elastic net, but implements a lasso regression instead
def NNLR_positive_Lasso(dataset_filtered, patterns_filtered, projectionName, alpha, iterations=10000):
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
    model = linear_model.Lasso(alpha=alpha, max_iter=iterations, positive=True)
    model.fit(patterns_filtered.X.T, dataset_filtered.X.T)
    dataset_filtered.obsm[projectionName] = model.coef_
    print(model.coef_.shape)


# still experimenting with this but same idea as NNLR_Elastic net, but implements ordinary least squares from scipy no
# regularization
def NNLR_LeastSquares(dataset_filtered, patterns_filtered, projectionName):
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
    print(patterns_filtered.X.shape[0], "patterns", dataset_filtered.X.T.shape[1], "data")
    pattern_matrix = np.zeros([patterns_filtered.X.shape[0], dataset_filtered.X.T.shape[1]])
    MSE = 0
    for i in range(dataset_filtered.shape[1]):
        scip = nnls(patterns_filtered.X.T, dataset_filtered.X.T[:, i])
        pattern_matrix[:, i] = scip[0]
        MSE = MSE + scip[1]
    print(pattern_matrix.shape)
    print(MSE, "Mean Squared Error")
    dataset_filtered.obsm[projectionName] = np.transpose(pattern_matrix)


# Parameters
# dataset_filtered : Anndata object cells x genes
# patterns_filtered : Anndata object features x genes
# cellTypeColumnName : String which is where the celltype data is stored
# num_cell_types : String the number of cell types in the dataset this parameter could be removed
# projectionName : String The name of the projection created using one of the regression matrices
# plotName : String the name of pearson matrix is the index for the pearson matrix in .uns this parameter could probably
# have a more informative name
# plot : boolean if True a plot is displayed
# Return-void
# Purpose of the Method:
# This method find the pearson correlation coefficient between every pattern and every cell type
def pearsonMatrix(dataset_filtered, patterns_filtered, cellTypeColumnName, num_cell_types, projectionName, plotName,
                  plot):
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
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
    if plot:
        pearsonViz(dataset_filtered, plotName)


# Parameters
# dataset_filtered : Anndata object cells x genes
# plotName : String index of pearson matrix to visualize
# Return -void
def pearsonViz(dataset_filtered, plotName):
    matcher.sourceIsValid(dataset_filtered)
    plt.title("Pearson Plot", fontsize=24)
    graphic = sns.heatmap(dataset_filtered.uns[plotName])
    plt.show()


# Parameters
# dataset_filtered : Anndata object cells x genes
# cellTypeColumnName : String index for cell type in dataset_filtered
# projectionName : String which is the name of the projection in obsm
# UMAPName : String index for the UMAP in dataset_filtered in obsm
# n_neighbors : int number of neighbors for the UMAP (side note: I understand what this parameter does, but I do not
# understand for instance how many neighbors to recommend) Another question might be like how do we know a UMAP is good
# metric : String the distance metric (I have not experimented with this, but I have heard that euclidean breaks down in
# higher dimensions. Maybe, trying some other metrics could be interesting.)
# plot : boolean for whether to plot the UMAP
# color : String for what color scheme to use this could become a potential issue with more and more cell types
# pointSize : double size of the points on the visualization only used if plot is True
# Return-void
# Purpose of the Method:
# This method performs an elastic net regression from Sklearn. The "discovered" pattern matrix is stored in
# the dataset_filtered.obsm under the paramater projectionName
def UMAP_Projection(dataset_filtered, cellTypeColumnName, projectionName, UMAPName, n_neighbors, metric='euclidean',
                    plot=True, color='Paired', pointSize=.5):
    matcher.sourceIsValid(dataset_filtered)
    color = matcher.mapCellNamesToInts(dataset_filtered, cellTypeColumnName)
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
    nd = umap_obj.fit_transform(dataset_filtered.obsm[projectionName])
    dataset_filtered.obsm[UMAPName] = nd
    if plot:
        UMAP_Viz(dataset_filtered, UMAPName, color, pointSize)


# Parameters
# dataset_filtered : Anndata object cells x genes
# UMAP Name : String index for the UMAP
# color : String color scheme
# pointSize : double for the size of the points
# Return-void
# Purpose of the Method:
# Visualize the UMAP of the "discovered" pattern matrix
def UMAP_Viz(dataset_filtered, UMAPName, color='Paired', pointSize=.5):
    matcher.sourceIsValid(dataset_filtered)
    nd = dataset_filtered.obsm[UMAPName]
    plt.scatter(nd[:, 0], nd[:, 1],
                c=[sns.color_palette("Paired", n_colors=12)[x] for x in color], s=pointSize)
    plt.title("UMAP Projection of Pattern Matrix", fontsize=24)
    plt.show()


# Parameters
# dataset_filtered : Anndata object cells x genes
# num_patterns : int the number of the patterns (this could probably be removed)
# projectionName : String which is the index of the projection in obsm
# UMAPName : String index of the UMAP in obsm
# Return-void
# Purpose of the Method:
# This creates plots showing the amount of each feature in each cell.
def featurePlots(dataset_filtered, num_patterns, projectionName, UMAPName):
    matcher.sourceIsValid(dataset_filtered)
    for i in range(num_patterns):
        pattern_matrix = dataset_filtered.obsm[projectionName]
        feature = pattern_matrix[:, i]
        plt.title("Feature " + str(i + 1), fontsize=24)
        plt.scatter(dataset_filtered.obsm[UMAPName][:, 0], dataset_filtered.obsm[UMAPName][:, 1], c=feature, cmap='jet',
                    s=.5)
        plt.colorbar()
        print("Number of nonzero cells " + str(np.count_nonzero(feature)))
        print("Percentage of nonzero cells " + str((np.count_nonzero(feature) / dataset_filtered.shape[0]) * 100))
        print("Max coefficient " + str(np.max(feature)))
        print("Average coefficient " + str(np.mean(feature)))
        plt.show()


# Parameters
# dataset_filtered : Anndata object cells x genes
# datasetFileName : String fileName
# Return - void
# Purpose of the method:
# Not sure if this is necessary, but it just saves the dataset with all of the new annotations
def saveProjections(dataset_filtered, datasetFileName):
    matcher.sourceIsValid(dataset_filtered)
    dataset_filtered.write.h5ad(datasetFileName)
