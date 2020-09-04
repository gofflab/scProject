import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import umap
from scipy import stats
from scipy.optimize import nnls, lsq_linear
from sklearn import linear_model
from . import matcher


def filterAnnDatas(dataset, patterns, geneColumnName):
    """ This method filters the patterns and the dataset to only include overlapping genes

    :param dataset: Anndata object cells x genes
    :type dataset: AnnData object
    :param patterns: Anndata object features x genes
    :param geneColumnName: index for where the gene names are kept in .var
    :return: A tuple of two filtered AnnData objects
    """

    matcher.sourceIsValid(dataset)  # Make sure dataset is an AnnData object
    matcher.sourceIsValid(patterns)  # Make sure patterns is an AnnData object
    dataset.var = dataset.var.set_index(geneColumnName)
    overlap = dataset.var.index.intersection(patterns.var.index)
    dataset_filtered = dataset[:, overlap]
    print(dataset_filtered.shape, "dataset filter shape")
    patterns_filtered = patterns[:, overlap]
    print(patterns_filtered.shape, "patterns filter shape")
    return dataset_filtered, patterns_filtered


def NNLR_ElasticNet(dataset_filtered, patterns_filtered, projectionName, alpha, L1, iterations=10000):
    """ This method performs an elastic net regression from sci-kit learn.

    :param dataset_filtered: AnnData object cells x genes
    :param patterns_filtered: AnnData object features x genes
    :param projectionName: index of the projection in dataset_filtered.obsm
    :type projectionName: String
    :param alpha: regularization parameter
    :type alpha: double
    :param L1: regularization parameter
    :type L1: double
    :param iterations: number of iterations while performing the regression
    :return: void, the dataset_filtered is mutated and the projection is stored in dataset_filtered.obsm[projectionName]
    """
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
    model = linear_model.ElasticNet(alpha=alpha, l1_ratio=L1, max_iter=iterations, positive=True)
    model.fit(patterns_filtered.X.T, dataset_filtered.X.T)
    dataset_filtered.obsm[projectionName] = model.coef_
    print(model.coef_.shape)


# still experimenting with this but same idea as NNLR_Elastic net, but implements a lasso regression instead
def NNLR_positive_Lasso(dataset_filtered, patterns_filtered, projectionName, alpha, iterations=10000):
    """This method performs a positive lasso regression from sci-kit learn.

    :param dataset_filtered: AnnData object cells x genes
    :param patterns_filtered: AnnData object features x genes
    :param projectionName: index of the projection in dataset_filtered.obsm
    :param alpha: regularization parameter
    :param iterations: number of iterations while performing the regression
    :return: void, the dataset_filtered is mutated and the projection is stored in dataset_filtered.obsm[projectionName]
    """
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
    model = linear_model.Lasso(alpha=alpha, max_iter=iterations, positive=True)
    model.fit(patterns_filtered.X.T, dataset_filtered.X.T)
    dataset_filtered.obsm[projectionName] = model.coef_
    print(model.coef_.shape)


# still experimenting with this but same idea as NNLR_Elastic net, but implements ordinary least squares from scipy no
# regularization
def NNLR_LeastSquares(dataset_filtered, patterns_filtered, projectionName):
    """This performs a non negative least squares regression using Scipy.

    :param dataset_filtered: AnnData object cells x genes
    :param patterns_filtered: AnnData object features x genes
    :param projectionName: index of the projection in dataset_filtered.obsm
    :return: void, the dataset_filtered is mutated and the projection is stored in dataset_filtered.obsm[projectionName]
    """
    matcher.sourceIsValid(dataset_filtered)
    matcher.sourceIsValid(patterns_filtered)
    print(patterns_filtered.X.shape[0], "patterns", dataset_filtered.X.T.shape[1], "data")
    pattern_matrix = np.zeros([patterns_filtered.X.shape[0], dataset_filtered.X.T.shape[1]])
    for i in range(dataset_filtered.shape[1]):
        scip = lsq_linear(np.array(patterns_filtered.X.T).astype('double'),
                          np.array(dataset_filtered.X.T[:, i]).astype('double'),
                          bounds=(0, 1000),
                          method='bvls',
                          lsq_solver='exact')
        pattern_matrix[:, i] = scip.x
    # print(pattern_matrix.shape)
    # print(MSE, "Mean Squared Error")
    dataset_filtered.obsm[projectionName] = np.transpose(pattern_matrix)


def pearsonMatrix(dataset_filtered, patterns_filtered, cellTypeColumnName, num_cell_types, projectionName, plotName,
                  plot):
    """This method find the pearson correlation coefficient between every pattern and every cell type

    :param dataset_filtered: Anndata object cells x genes
    :param patterns_filtered: Anndata object features x genes
    :param cellTypeColumnName: index where the cell types are stored in dataset_filtered.obsm
    :param num_cell_types: The number of cell types in the dataset this parameter could be removed
    :param projectionName: The name of the projection created using one of the regression methods
    :param plotName: The index for the pearson matrix in dataset_filtered.uns[plotName]
    :param plot: If True a plot is displayed
    :type plot: boolean
    :type projectionName: String
    :type num_cell_types: int
    :type cellTypeColumnName: String
    :return: void
    """
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
        pearsonViz(dataset_filtered, plotName, cellTypeColumnName)


def pearsonViz(dataset_filtered, plotName, cellTypeColumnName):
    """ Visualize a Pearson Matrix.

    :param dataset_filtered: Anndata object cells x genes
    :param plotName: Index of pearson matrix to visualize
    :param cellTypeColumnName: index for cell type in dataset_filtered.obsm
    :return: void
    """
    matcher.sourceIsValid(dataset_filtered)
    y_ticks = []
    for i in range(dataset_filtered.uns[plotName].shape[0]):
        y_ticks.append('Feature ' + str(i+1))

    plt.title("Pearson Plot", fontsize=24)
    sns.heatmap(dataset_filtered.uns[plotName],
                xticklabels=dataset_filtered.obs[cellTypeColumnName].unique(),
                yticklabels=y_ticks)
    plt.tick_params(axis='y', labelsize=4)
    plt.show()


# n_neighbors : int number of neighbors for the UMAP (side note: I understand what this parameter does, but I do not
# understand for instance how many neighbors to recommend) Another question might be like how do we know a UMAP is good
# metric : String the distance metric (I have not experimented with this, but I have heard that euclidean breaks down in
# higher dimensions. Maybe, trying some other metrics could be interesting.)

def UMAP_Projection(dataset_filtered, cellTypeColumnName, projectionName, UMAPName, n_neighbors, metric='euclidean',
                    plot=True, colorScheme='Paired', pointSize=.5):
    """This method projects the pattern matrix down into 2 dimensional space. Make sure the colorScheme chosen has
    enough colors for every cell type.

    :param dataset_filtered: Anndata object cells x genes
    :param cellTypeColumnName: index for cell type in dataset_filtered.obsm
    :param projectionName: index for the projection in dataset_filtered.obsm
    :param UMAPName: index for the created UMAP coordinates in dataset_filtered.obsm
    :param n_neighbors: number of neighbors for the UMAP
    :param metric: the distance metric used in the UMAP, defaults to euclidean
    :param plot: If True a plot is displayed, defaults to True
    :param colorScheme: seaborn color scheme to use, defaults to Paired
    :param pointSize: size of the points, defaults to .5
    :return: void, mutates dataset_filtered and add the UMAP to obsm
    """
    matcher.sourceIsValid(dataset_filtered)
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
    nd = umap_obj.fit_transform(dataset_filtered.obsm[projectionName])
    dataset_filtered.obsm[UMAPName] = nd
    if plot:
        UMAP_Viz(dataset_filtered, UMAPName, cellTypeColumnName, colorScheme, pointSize)


def UMAP_Viz(dataset_filtered, UMAPName, cellTypeColumnName, colorScheme='Paired', pointSize=.5):
    """Plots the UMAP of the pattern matrix. Make sure colorScheme has at least as many colors as cell types in your
    dataset.

    :param cellTypeColumnName: index for cell type in dataset_filtered.obsm
    :param dataset_filtered: Anndata object cells x genes
    :param UMAPName: index for the UMAP in dataset_filtered.obsm
    :param colorScheme: seaborn color scheme, defaults to Paired
    :param pointSize: size of the points, defaults to .5
    :return: void
    """
    matcher.sourceIsValid(dataset_filtered)
    color = matcher.mapCellNamesToInts(dataset_filtered, cellTypeColumnName)
    nd = dataset_filtered.obsm[UMAPName]
    plt.scatter(nd[:, 0], nd[:, 1],
                c=[sns.color_palette(colorScheme,
                                     n_colors=dataset_filtered.obs[cellTypeColumnName].unique().shape[0])[x] for x in
                   color], s=pointSize)
    plt.title("UMAP Projection of Pattern Matrix", fontsize=24)
    handles = []
    for i in range(dataset_filtered.obs[cellTypeColumnName].unique().shape[0]):
        handles.append(mpatches.Patch(color=(sns.color_palette(colorScheme, n_colors=12)[i]),
                                      label=dataset_filtered.obs[cellTypeColumnName].unique()[i]))
    plt.legend(handles=handles, title="Cell Types", fontsize='xx-small', loc='best')
    plt.show()


def featurePlots(dataset_filtered, num_patterns, projectionName, UMAPName, cmap='viridis'):
    """Creates plots which show the weight of each feature in each cell.

    :param dataset_filtered: Anndata object cells x genes
    :param num_patterns: the number of the patterns to display starting from feature 1. It can also take a list of ints.
    :param projectionName: index of the projection in dataset_filtered.obsm
    :param UMAPName: index of the UMAP in dataset_filtered.obsm
    :param cmap: colormap to use when creating the feature plots
    :return: void
    """
    matcher.sourceIsValid(dataset_filtered)
    if isinstance(num_patterns, list):
        for i in num_patterns:
            pattern_matrix = dataset_filtered.obsm[projectionName]
            feature = pattern_matrix[:, i-1]
            plt.title("Feature " + str(i), fontsize=24)
            plt.scatter(dataset_filtered.obsm[UMAPName][:, 0], dataset_filtered.obsm[UMAPName][:, 1], c=feature,
                        cmap=cmap,
                        s=.5)
            plt.colorbar()
            print("Number of nonzero cells " + str(np.count_nonzero(feature)))
            print("Percentage of nonzero cells " + str((np.count_nonzero(feature) / dataset_filtered.shape[0]) * 100))
            print("Max coefficient " + str(np.max(feature)))
            print("Average coefficient " + str(np.mean(feature)))
            plt.show()
    else:
        for i in range(num_patterns):
            pattern_matrix = dataset_filtered.obsm[projectionName]
            feature = pattern_matrix[:, i]
            plt.title("Feature " + str(i + 1), fontsize=24)
            plt.scatter(dataset_filtered.obsm[UMAPName][:, 0], dataset_filtered.obsm[UMAPName][:, 1], c=feature,
                        cmap=cmap,
                        s=.5)
            plt.colorbar()
            print("Number of nonzero cells " + str(np.count_nonzero(feature)))
            print("Percentage of nonzero cells " + str((np.count_nonzero(feature) / dataset_filtered.shape[0]) * 100))
            print("Max coefficient " + str(np.max(feature)))
            print("Average coefficient " + str(np.mean(feature)))
            plt.show()


def featureImportance(dataset_filtered, num_patterns, projectionName):
    """Shows a bar graph of feature importance/usage as measured by average coefficient.

    :param dataset_filtered: Anndata object cells x genes
    :param num_patterns: the number of the patterns to display starting from feature 1. It can also take a list of ints.
    :param projectionName: index of the projection in dataset_filtered.obsm
    :return:
    """
    matcher.sourceIsValid(dataset_filtered)
    if isinstance(num_patterns, list):
        importance = []
        for i in num_patterns:
            pattern_matrix = dataset_filtered.obsm[projectionName]
            feature = pattern_matrix[:, i]
            importance.append(np.mean(feature))
        plt.bar(num_patterns, importance, tick_label=num_patterns)
        plt.xlabel('Features')
        plt.ylabel('Avg. Coefficient')
        plt.title('Feature Importance as ranked by avg. coefficient')
        plt.show()

    else:
        importance2 = []
        for i in range(num_patterns):
            pattern_matrix = dataset_filtered.obsm[projectionName]
            feature = pattern_matrix[:, i]
            importance2.append(np.mean(feature))
        plt.bar(range(1, num_patterns+1), importance2, tick_label=range(1, num_patterns+1))
        plt.tick_params(axis='x', labelsize=6)
        plt.xlabel('Features')
        plt.ylabel('Avg. Coefficient')
        plt.title('Feature Importance as ranked by avg. coefficient', fontsize=24)
        plt.show()


def importantGenes(patterns_filtered, featureNumber, threshold):
    """Returns the list of genes that are expressed greater than threshold in the feature.

    :param patterns_filtered: AnnData object features x genes
    :param featureNumber: Which pattern you want to examine
    :param threshold: Show genes that are greater than this threshold
    :return: A list of genes that are expressed above threshold in the pattern
    """
    where = np.where(patterns_filtered.X.T[:, featureNumber-1] > threshold)
    ids = [patterns_filtered.var.index[i] for i in where]
    return ids


def saveProjections(dataset_filtered, datasetFileName):
    """Allows user to save the mutated dataset after analysis

    :param dataset_filtered: AnnData object cells x genes
    :param datasetFileName: the filename for the new object
    :return: void
    """
    matcher.sourceIsValid(dataset_filtered)
    dataset_filtered.write.h5ad(datasetFileName)
