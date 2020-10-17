from . import matcher
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import umap
import seaborn as sns
import matplotlib.patches as mpatches


def pearsonMatrix(dataset_filtered, patterns_filtered, cellTypeColumnName, num_cell_types, projectionName, plotName,
                  plot, row_cluster=True):
    """This method find the pearson correlation coefficient between every pattern and every cell type

    :param row_cluster: Bool whether to cluster
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
        pearsonViz(dataset_filtered, plotName, cellTypeColumnName, row_cluster)


def pearsonViz(dataset_filtered, plotName, cellTypeColumnName, row_cluster=True):
    """ Visualize a Pearson Matrix.


    :param dataset_filtered: Anndata object cells x genes
    :param plotName: Index of pearson matrix to visualize
    :param cellTypeColumnName: index for cell type in dataset_filtered.obsm
    :param row_cluster: Bool whether to cluster rows or not
    :return: void
    """
    matcher.sourceIsValid(dataset_filtered)
    y_ticks = []
    for i in range(dataset_filtered.uns[plotName].shape[0]):
        y_ticks.append('Feature ' + str(i + 1))

    # plt.title("Pearson Plot", fontsize=24)
    if row_cluster:
        sns.set(font_scale=1)
        cluster = sns.clustermap(dataset_filtered.uns[plotName],
                                 row_cluster=row_cluster,
                                 col_cluster=False,
                                 xticklabels=dataset_filtered.obs[cellTypeColumnName].unique(),
                                 yticklabels=y_ticks)
        cluster.ax_heatmap.set_yticklabels(cluster.ax_heatmap.yaxis.get_majorticklabels(), fontsize=5)
        # cluster.set_yticklabels(cluster.get_yticks(), size=10)
        # plt.tick_params(axis='y', labelsize=15)
    else:
        plt.title("Pearson Plot", fontsize=24)
        sns.heatmap(dataset_filtered.uns[plotName],
                    xticklabels=dataset_filtered.obs[cellTypeColumnName].unique(),
                    yticklabels=y_ticks)
        plt.tick_params(axis='y', labelsize=4)
    plt.show()


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
            feature = pattern_matrix[:, i - 1]
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
