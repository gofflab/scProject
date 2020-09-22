import numpy as np
import matplotlib.pyplot as plt
from . import matcher


def importantGenes(patterns_filtered, featureNumber, threshold):
    """Returns the list of genes that are expressed greater than threshold in the feature.

    :param patterns_filtered: AnnData object features x genes
    :param featureNumber: Which pattern you want to examine
    :param threshold: Show genes that are greater than this threshold
    :return: A list of genes that are expressed above threshold in the pattern
    """
    where = np.where(patterns_filtered.X.T[:, featureNumber - 1] > threshold)
    ids = [patterns_filtered.var.index[i] for i in where]
    return ids


def geneSelectivity(patterns_filtered, geneName, num_pattern, plot=True):
    """Computes the percentage of a genes expression in a feature out of the total gene expression over all features.

    :param patterns_filtered: AnnData object features x genes
    :param geneName: geneName must be in the format that is in .var
    :param num_pattern: The number of the feature/pattern of interest
    :param plot: boolean, if true plots expression of the gene across all of the patterns.
    :return:
    """

    array = []
    for i in range(patterns_filtered.shape[1]):
        array.append(patterns_filtered.var.index[i])
    array = np.array(array)
    tuple1 = np.where(array == geneName)[0]
    index = tuple1[0]
    geneRow = patterns_filtered.X.T[index]  # gene in question
    sumRow = np.sum(geneRow)
    percent = (geneRow[num_pattern - 1] / sumRow) * 100
    print("Feature " + str(num_pattern) + " expresses " + str(percent) + "% of gene " + str(
        patterns_filtered.var.index[index]))
    if plot:
        plt.title("Expression of " + str(patterns_filtered.var.index[index]) + " across all features")
        plt.bar(np.arange(1, patterns_filtered.shape[0] + 1), geneRow)
        plt.xticks(np.arange(1, patterns_filtered.shape[0] + 1), patterns_filtered.obs.index, rotation=90)
        plt.tick_params(axis='x', labelsize=6)
        plt.show()


def geneDriver(dataset_filtered, patterns_filtered, geneName, cellTypeColumnName, cellType, projectionName):
    """

    :param dataset_filtered: Anndata object cells x genes
    :param patterns_filtered: AnnData object features x genes
    :param geneName: Name of gene in question must be in .var
    :param cellTypeColumnName: index for cell type in dataset_filtered.obsm
    :param cellType: str celltype in question
    :param projectionName: str projection from which to use the pattern weights
    :return:
    """
    array = []
    for i in range(patterns_filtered.shape[1]):
        array.append(patterns_filtered.var.index[i])
    array = np.array(array)
    tuple1 = np.where(array == geneName)[0]
    index = tuple1[0]
    geneRow = np.transpose(patterns_filtered.X.T[index]) / np.sum(patterns_filtered.X.T[index])
    cells = dataset_filtered.obs[cellTypeColumnName]
    where = np.where(cells == cellType)
    cellTypePatterns = dataset_filtered.obsm[projectionName][where]
    print(cellTypePatterns.shape)
    avgPWeights = np.sum(cellTypePatterns, axis=0) / cellTypePatterns.shape[0]
    stat = np.multiply(geneRow, avgPWeights)
    plt.bar(np.arange(1, patterns_filtered.shape[0] + 1), stat)
    plt.title("Features that drive the expression of " + geneName + " in " + cellType)
    plt.xticks(np.arange(1, patterns_filtered.shape[0] + 1), patterns_filtered.obs.index, rotation=90)
    plt.tick_params(axis='x', labelsize=6)
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
        plt.bar(range(1, num_patterns + 1), importance2, tick_label=range(1, num_patterns + 1))
        plt.tick_params(axis='x', labelsize=6)
        plt.xlabel('Features')
        plt.ylabel('Avg. Coefficient')
        plt.title('Feature Importance as ranked by avg. coefficient', fontsize=24)
        plt.show()
