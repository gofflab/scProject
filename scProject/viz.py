from . import matcher
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from scipy import stats
import numpy as np
import umap
import seaborn as sns
import matplotlib.patches as mpatches


def pearsonMatrix(dataset_filtered, patterns_filtered, cellTypeColumnName, num_cell_types, projectionName, plotName,
                  plot, row_cluster=True, col_cluster=True, path=None, display=True, dpi=300, xtickSize=8, ytickSize=8):
    """This method finds the pearson correlation coefficient between every pattern and every cell type

    :param dataset_filtered: Anndata object cells x genes
    :param patterns_filtered: Anndata object features x genes
    :param cellTypeColumnName: index where the cell types are stored in dataset_filtered.obsm
    :param num_cell_types: The number of cell types in the dataset this parameter could be removed
    :param projectionName: The name of the projection created using one of the regression methods
    :param plotName: The index for the pearson matrix in dataset_filtered.uns[plotName]
    :param plot: If True a plot is generated either saved or displayed
    :param row_cluster: Bool whether to cluster
    :param col_cluster: Bool whether to cluster columns or not
    :param dpi: Quality of image to be saved
    :param display: Bool whether to display the plot or not
    :param xtickSize: Size of labels on the x-axis
    :param ytickSize: Size of labels on the y-axis
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
        pearsonViz(dataset_filtered, plotName, cellTypeColumnName, row_cluster, col_cluster, path, display, dpi,
                   xtickSize, ytickSize)


def pearsonViz(dataset_filtered, plotName, cellTypeColumnName, row_cluster=True, col_cluster=True, path=None,
               display=True, dpi=300, xtickSize=8, ytickSize=8):
    """ Visualize or save a Pearson Matrix.

    :param path:
    :param dataset_filtered: Anndata object cells x genes
    :param plotName: Index of pearson matrix to visualize
    :param cellTypeColumnName: index for cell type in dataset_filtered.obsm
    :param row_cluster: Bool whether to cluster rows or not
    :param col_cluster: Bool whether to cluster columns or not
    :param dpi: Quality of image to be saved
    :param display: Bool whether to display the plot or not
    :param xtickSize: Size of labels on the x-axis
    :param ytickSize: Size of labels on the y-axis

    :return: void
    """
    matcher.sourceIsValid(dataset_filtered)
    y_ticks = []
    for i in range(dataset_filtered.uns[plotName].shape[0]):
        y_ticks.append('Feature ' + str(i + 1))

    if row_cluster or col_cluster:
        sns.set(font_scale=1)
        cluster = sns.clustermap(dataset_filtered.uns[plotName],
                                 row_cluster=row_cluster,
                                 col_cluster=col_cluster,
                                 xticklabels=dataset_filtered.obs[cellTypeColumnName].unique(),
                                 yticklabels=y_ticks)
        cluster.ax_heatmap.set_yticklabels(cluster.ax_heatmap.yaxis.get_majorticklabels(), fontsize=ytickSize)
        cluster.ax_heatmap.set_xticklabels(cluster.ax_heatmap.xaxis.get_majorticklabels(), fontsize=xtickSize)
    else:
        plt.title("Pearson Plot", fontsize=24)
        sns.heatmap(dataset_filtered.uns[plotName],
                    xticklabels=dataset_filtered.obs[cellTypeColumnName].unique(),
                    yticklabels=y_ticks)
        plt.tick_params(axis='y', labelsize=ytickSize)
        plt.tick_params(axis='x', labelsize=xtickSize)
    if path is None and display is True:
        plt.show()
    if path is not None and display is False:
        plt.savefig(path, dpi=dpi)
        plt.close()
    if path is not None and display is True:
        plt.show()
        plt.savefig(path, dpi=dpi)


def UMAP_Projection(dataset_filtered, cellTypeColumnName, projectionName, UMAPName, n_neighbors, metric='euclidean',
                    plot=True, colorScheme='Paired', pointSize=.5, subset=None, path=None, display=True, dpi=300):
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
    :param subset: subset of types in cell type column name to plot
    :param path: path to save figure
    :param display: Whether to display the figure in the jupyter notebook
    :param dpi: Quality of the plot to be saved
    :return: void, mutates dataset_filtered and add the UMAP to obsm
    """
    matcher.sourceIsValid(dataset_filtered)
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
    nd = umap_obj.fit_transform(dataset_filtered.obsm[projectionName])
    dataset_filtered.obsm[UMAPName] = nd
    if plot:
        UMAP_Viz(dataset_filtered, UMAPName, cellTypeColumnName, colorScheme, pointSize, subset, path, display, dpi)


def UMAP_Viz(dataset_filtered, UMAPName, cellTypeColumnName, colorScheme='Paired', pointSize=.5, subset=None, path=None,
             display=True, dpi=300):
    """Plots the UMAP of the pattern matrix. Make sure colorScheme has at least as many colors as cell types in your
    dataset.

    :param cellTypeColumnName: index for cell type in dataset_filtered.obs can be any column in .obs
    :param dataset_filtered: Anndata object cells x genes
    :param UMAPName: index for the UMAP in dataset_filtered.obsm
    :param colorScheme: seaborn color scheme, defaults to Paired
    :param pointSize: size of the points, defaults to .5
    :param subset: subset of types in cell type column name to plot
    :param path: path to save figure
    :param display: Whether to display the figure in the jupyter notebook
    :param dpi: Quality of the plot to be saved
    :return: void
    """
    if subset is None:
        matcher.sourceIsValid(dataset_filtered)
        color = matcher.mapCellNamesToInts(dataset_filtered, cellTypeColumnName)
        numTypes = dataset_filtered.obs[cellTypeColumnName].unique().shape[0]
        colors = []
        palette = sns.color_palette(colorScheme, n_colors=numTypes)
        for x in range(color.shape[0]):
            colors.append(palette[color[x]])
        nd = dataset_filtered.obsm[UMAPName]
        plt.scatter(nd[:, 0], nd[:, 1],
                    c=colors, s=pointSize, rasterized=True)
        plt.title("UMAP Projection of Pattern Matrix", fontsize=24)
        handles = []
        for i in range(dataset_filtered.obs[cellTypeColumnName].unique().shape[0]):
            handles.append(mpatches.Patch(color=(sns.color_palette(colorScheme, n_colors=numTypes)[i]),
                                          label=dataset_filtered.obs[cellTypeColumnName].unique()[i]))
        plt.legend(handles=handles, title="Cell Types", fontsize='xx-small', loc='best')
        if display is True and path is None:
            plt.show()
        if display is True and path is not None:
            plt.show()
            plt.savefig(path, dpi=dpi)
        if display is False and path is not None:
            plt.savefig(path, dpi=dpi)
            plt.close()
    else:
        subsetted = dataset_filtered[dataset_filtered.obs[cellTypeColumnName].isin(subset)]
        color = matcher.mapCellNamesToInts(subsetted, cellTypeColumnName)
        nd = subsetted.obsm[UMAPName]
        numTypes = subsetted.obs[cellTypeColumnName].unique().shape[0]
        colors = []
        palette = sns.color_palette(colorScheme, n_colors=numTypes)
        for x in range(color.shape[0]):
            colors.append(palette[color[x]])
        plt.scatter(nd[:, 0], nd[:, 1],
                    c=colors, s=pointSize, rasterized=True)
        plt.title("UMAP Projection of Pattern Matrix", fontsize=24)
        handles = []
        for i in range(numTypes):
            handles.append(mpatches.Patch(color=(sns.color_palette(colorScheme, n_colors=numTypes)[i]),
                                          label=subsetted.obs[cellTypeColumnName].unique()[i]))
        plt.legend(handles=handles, title="Cell Types", fontsize='xx-small', loc='best')
        if display is True and path is None:
            plt.show()
        if display is True and path is not None:
            plt.show()
            plt.savefig(path, dpi=dpi)
        if display is False and path is not None:
            plt.savefig(path, dpi=dpi)
            plt.close()


def featurePlots(dataset_filtered, num_patterns, projectionName, UMAPName, vmin=.00000000001, clip=99.5,
                 zeroColor='dimgrey', obsColumn=None, cmap='viridis', pointSize=.1, subset=None,
                 path=None, display=True, dpi=300):
    """Creates plots which show the weight of each feature in each cell.

    :param clip: Stops colorbar at the percentile specified [0,100]
    :param vmin: Min of the colorplot i.e. what to define as zero
    :param zeroColor: What color the cells below vmin should be colored
    :param dataset_filtered: Anndata object cells x genes
    :param num_patterns: the number of the patterns to display starting from feature 1. It can also take a list of ints.
    :param projectionName: index of the projection in dataset_filtered.obsm
    :param UMAPName: index of the UMAP in dataset_filtered.obsm
    :param obsColumn: Column in dataset_filtered to use for subsetting
    :param cmap: colormap to use when creating the feature plots
    :param pointSize: Size of the points on the plots
    :param subset: subset of types in cell type column name to plot
    :param path: path to save figure without a file type suffix like pdf png
    :param display: Whether to display the figure in the jupyter notebook
    :param dpi: Quality of the plot to be saved
    :return: void, files will be .png
    """
    matcher.sourceIsValid(dataset_filtered)
    if subset is not None:
        if obsColumn is None:
            raise ValueError("obsColum cannot be None when filtering")
        data = dataset_filtered[dataset_filtered.obs[obsColumn].isin(subset)]
        pattern_matrix = data.obsm[projectionName]
        print(pattern_matrix.shape)
    if subset is None:
        pattern_matrix = dataset_filtered.obsm[projectionName]
        data = dataset_filtered
    if isinstance(num_patterns, list):
        colors = plt.get_cmap(cmap)
        colors.set_under(zeroColor)
        for i in num_patterns:
            feature = pattern_matrix[:, i - 1]
            plt.title("Feature " + str(i), fontsize=24)
            plt.scatter(data.obsm[UMAPName][:, 0], data.obsm[UMAPName][:, 1], c=feature,
                        cmap=colors, vmin=vmin, vmax=np.percentile(feature, clip), s=pointSize)
            plt.colorbar()
            print("Number of nonzero cells " + str(np.count_nonzero(feature)))
            if path is None:
                plt.show()
            if path is not None and display is True:
                plt.show()
                plt.savefig(path + str(i) + ".png", dpi=dpi)
            if path is not None and display is False:
                plt.savefig(path + str(i) + ".png", dpi=dpi)
                plt.close()
    else:
        colors = plt.get_cmap(cmap)
        colors.set_under(zeroColor)
        for i in range(num_patterns):
            feature = pattern_matrix[:, i]
            plt.title("Feature " + str(i + 1), fontsize=24)
            plt.scatter(data.obsm[UMAPName][:, 0], data.obsm[UMAPName][:, 1], c=feature, cmap=colors, vmin=vmin,
                        vmax=np.percentile(feature, clip), s=pointSize)
            plt.colorbar()
            print("Number of nonzero cells " + str(np.count_nonzero(feature)))
            if path is None:
                plt.show()
            if path is not None and display is True:
                plt.show()
                plt.savefig(path + str(i + 1) + ".png", dpi=dpi)
            if path is not None and display is False:
                plt.savefig(path + str(i + 1) + ".png", dpi=dpi)
                plt.close()


def patternWeightDistribution(dataset_filtered, projectionName, patterns, obsColumn, subset, numBins=100):
    """

    :param dataset_filtered: Anndata object cells x genes
    :param projectionName:index of the projection in dataset_filtered.obsm
    :param patterns: Which patterns to visualize (one indexed)
    :param obsColumn: Column in dataset_filtered to use for subsetting
    :param subset: What subset of cells in the obsColumn to visualize
    :param numBins: How many bins in the histogram
    :return: void displays a histogram of the pattern weights above 0
    """
    subset = dataset_filtered[dataset_filtered.obs[obsColumn].isin(list(subset))]
    print("This subset has shape:", subset.shape)

    for i in patterns:
        filte = subset.obsm[projectionName][:, i-1] > 0
        maxval = np.max(subset.obsm[projectionName][:, i-1][filte])
        if maxval is 0:
            print("Feature " + str(i) + " is all 0 in this subset")
            continue
        bins = np.arange(0, maxval+1, (maxval + 1) / numBins)

        plt.title("Feature " + str(i))
        plt.hist(subset.obsm[projectionName][:, i-1][filte], bins=bins)
        plt.show()
