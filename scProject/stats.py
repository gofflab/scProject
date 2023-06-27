import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from . import matcher
from scipy.stats import t
from scipy.sparse import issparse
import math
import pandas as pd
import os


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
    :return: void
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
    :return: void
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
    :return: void
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


def HotellingT2(cluster1, cluster2):
    """Calculates Hotelling T2 statistic to evaluate significance of difference means using pooled covariance and pseudo-inverse.
    
    :param cluster1: Anndata with cluster 1
    :param cluster2: Anddata with cluster 2
    :return: Tuple of F value, TSquared, p value
    """
    if (cluster1.shape[0] == 0) or (cluster2.shape[0] == 0):
        raise ValueError("Empty Cluster")
    if cluster1.shape[1] != cluster2.shape[1]:
        raise ValueError("Cluster1 and Cluster2 must have the same number of features (genes) i.e. same shape[1]")
    dimensionality = cluster1.shape[1]  # of genes = num features = dimensions
    n1 = cluster1.shape[0]
    n2 = cluster2.shape[0]

    if n1 + n2 - 2 < dimensionality:
        raise ValueError("The Hotelling T2 squared statistic is not applicable when the sum of the sample sizes minus "
                         "2 is less than the dimensionality.")

    if scipy.sparse.issparse(cluster1.X):
        C1Mean = cluster1.X.mean(axis=0)
    else:
        C1Mean = np.mean(cluster1.X, axis=0, keepdims=True)

    if scipy.sparse.issparse(cluster2.X):
        C2Mean = cluster2.X.mean(axis=0)
    else:
        C2Mean = np.mean(cluster2.X, axis=0, keepdims=True)

    MeanDiff = (C1Mean - C2Mean)  # *patterns_filtered.X[34, :]

    # covariance matrices and transform to pooled
    C1CovM = np.cov(cluster1.X, rowvar=False, bias=False)
    C2CovM = np.cov(cluster2.X, rowvar=False, bias=False)
    pooled = ((n1 - 1) * C1CovM + (n2 - 1) * C2CovM) / (n1 + n2 - 2)
    inverse = np.linalg.pinv(pooled * ((1 / n1) + (1 / n2)))

    step1 = np.matmul(MeanDiff, inverse)
    TSquared = np.matmul(step1, np.transpose(MeanDiff))

    Fval = ((n1 + n2 - dimensionality - 1) / (dimensionality * (n1 + n2 - 2))) * TSquared
    F = scipy.stats.f(dimensionality, n1 + n2 - dimensionality - 1)
    p_value = 1 - F.cdf(Fval)
    print("T2 Value: " + str(TSquared[0][0]) + " FValue: " + str(Fval[0][0]) + " P-Value: " + str(p_value[0][0]))
    return TSquared, Fval, p_value


def featureExpressionSig(cluster1, projectionName, featureNumber, alpha, mu=0):
    """Measure the significance of the mean expression of a feature for a group of cells.

    :param cluster1: AnnData with group of cells in question
    :param projectionName: Regression to use
    :param featureNumber: feature-value to use one-index
    :param alpha: Level of significance
    :return: tuple T and t value at alpha and degrees of freedom cells minus 1
    """
    pattern_weights = cluster1.obsm[projectionName][:, featureNumber - 1]
    mean = np.mean(pattern_weights)
    T = (mean - mu) / (np.std(pattern_weights) / math.sqrt(pattern_weights.shape[0]))
    testT = t.ppf(q=alpha, df=(pattern_weights.shape[0] - 1))
    if T > testT:
        print("We can reject the null hypothesis that the mean is " + str(mu) + ".")
    else:
        print("We cannot reject the null hypothesis that the mean is " + str(mu) + ".")
    return T, testT


def BonferroniCorrectedDifferenceMeans(cluster1, cluster2, alpha, varName, verbose=0, display=True):
    """ Constructs Bonferroni corrected Confidence intervals for the difference of the means of cluster1 and cluster2

       :param display: Whether to display CI visualization
       :param verbose: Whether to print out genes and CIs or not. 0 for print out. !=0 for no printing.
       :param cluster1: Anndata containing cluster1 cells
       :param cluster2: Anndata containing cluster2 cells
       :param alpha: Confidence value
       :param varName: What column in var to use for gene names
       :return: A dataframe of genes as index with their respective confidence intervals in columns low and high.
       """
    dimensionality = cluster1.shape[1]
    # print("weird")

    if issparse(cluster1.X):
        print("C1 is sparse")
        C1Mean = cluster1.X.mean(axis=0)
        C1Dense = cluster1.X.todense()
    else:
        print("C1 is dense")
        C1Mean = np.mean(cluster1.X, axis=0, keepdims=True)

    if issparse(cluster2.X):
        print("C2 is sparse")
        C2Mean = cluster2.X.mean(axis=0)
        C2Dense = cluster2.X.todense()
    else:
        print("C2 is dense")
        C2Mean = np.mean(cluster2.X, axis=0, keepdims=True)

    # C1Mean = np.mean(cluster1.X, axis=0, keepdims=True)
    # C2Mean = np.mean(cluster2.X, axis=0, keepdims=True)
    MeanDiff = (C1Mean - C2Mean)
    plusminus = np.zeros((2, MeanDiff.shape[1]))
    n1 = cluster1.shape[0]
    n2 = cluster2.shape[0]
    pVal = 1 - alpha
    boncorrect = pVal / (2 * dimensionality)
    q = 1 - boncorrect

    tval = t.ppf(q=q, df=n1 + n2 - 2)
    if issparse(cluster1.X):
        C1CovM = np.cov(C1Dense, rowvar=False, bias=False)
    else:
        C1CovM = np.cov(cluster1.X, rowvar=False, bias=False)
    if issparse(cluster2.X):
        C2CovM = np.cov(C2Dense, rowvar=False, bias=False)
    else:
        C2CovM = np.cov(cluster2.X, rowvar=False, bias=False)

    pooled = ((n1 - 1) * C1CovM + (n2 - 1) * C2CovM) / (n1 + n2 - 2)

    list1 = []

    for i in range(dimensionality):
        pooledVar = pooled[i, i]
        scale = tval * math.sqrt(((1 / n1) + (1 / n2)) * pooledVar)

        diff = MeanDiff[0, i]
        plusminus[0, i] = diff - scale
        plusminus[1, i] = diff + scale
        if plusminus[0, i] < 0 and plusminus[1, i] < 0:
            if verbose == 0:
                print('Gene Name: {0} Mean Difference: {1} Low: {2} High {3}'.format(cluster1.var[varName][i], diff,
                                                                                     plusminus[0, i], plusminus[1, i]))
            gene = cluster1.var[varName][i]
            list1.append(gene)

        if plusminus[0, i] > 0 and plusminus[1, i] > 0:
            if verbose == 0:
                print('Gene Name: {0} Mean Difference: {1} Low: {2} High {3}'.format(cluster1.var[varName][i], diff,
                                                                                     plusminus[0, i], plusminus[1, i]))
            gene = cluster1.var[varName][i]
            list1.append(gene)

    data = pd.DataFrame(np.transpose(plusminus), index=cluster1.var[varName], columns=['Low', 'High'])
    filt = ((data['High'] > 0) & (data['Low'] > 0)) | ((data['High'] < 0) & (data['Low'] < 0))
    intersection = data[filt]
    if display:
        right = intersection[(intersection['High'] > 0) & (intersection['Low'] > 0)]
        left = intersection[(intersection['High'] < 0) & (intersection['Low'] < 0)]

        for low, high, y in zip(right['Low'], right['High'], range(len(right))):
            plt.plot((low, high), (y, y), 'o-', color='blue')

        for low, high, y in zip(left['Low'], left['High'], range(len(right), len(intersection))):
            plt.plot((low, high), (y, y), 'o-', color='red')

        plt.yticks(range(len(intersection)), list(right.index) + list(left.index))
        plt.show()

    return data


def projectionDriver(patterns_filtered, cluster1, cluster2, alpha, varName, featureNumber, display=True,
                     num_annotated=25, path=None, titleFontsize=25, axisFontsize=25, pointSize=3, annotationSize=25):
    """Assumes patterns_filtered and dataset_filtered have the same index.

    :param patterns_filtered: AnnData object features x genes
    :param cluster1: Anndata containing cluster1 cells
    :param cluster2: Anndata containing cluster2 cells
    :param alpha: Confidence value for the bonferroni confidence intervals
    :param varName: What column in var to use for gene names
    :param featureNumber: Which pattern you want to examine
    :param display: Whether to visualize or not
    :param num_annotated: number of genes to annotate
    :param path: Path to save the visualization
    :param titleFontsize: fontsize of the title
    :param axisFontsize: fontsize of the axis labels
    :param pointSize: size of the points
    :param annotationSize: Size of the annotations for the top ranked genes
    :return: Three dataframes. The first is the significant gene drivers. The second is Bonferroni CIs from the weighted mean vector. The third is the standard bonferroni CIs equivalent to calling BonferroniCorrectedDifferenceMeans.
    """

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

    if (featureNumber > patterns_filtered.shape[0]) or featureNumber < 0:
        raise ValueError("Invalid feature number. Must be between 0 and " + str(patterns_filtered.shape[0] + "."))

    dimensionality = cluster1.shape[1]
    if scipy.sparse.issparse(cluster1.X):
        C1Mean = cluster1.X.mean(axis=0)
        C1Dense = cluster1.X.todense()
    else:
        C1Mean = np.mean(cluster1.X, axis=0, keepdims=True)

    if scipy.sparse.issparse(cluster2.X):
        C2Mean = cluster2.X.mean(axis=0)
        C2Dense = cluster2.X.todense()
    else:
        C2Mean = np.mean(cluster2.X, axis=0, keepdims=True)

    # C1Mean = np.mean(cluster1.X, axis=0, keepdims=True)
    # C2Mean = np.mean(cluster2.X, axis=0, keepdims=True)

    MeanDiff = (C1Mean - C2Mean)
    feature = patterns_filtered[(featureNumber - 1)].copy()
    # construct weighted Mean Difference using specified feature
    curNorm = np.linalg.norm(np.reshape(feature.X, -1), ord=1)
    numNonzero = np.count_nonzero(feature.X)
    normalized = feature.X * (numNonzero / curNorm)
    drivers = np.multiply(np.reshape(normalized, -1), np.reshape(MeanDiff, -1))
    drivers = np.reshape(drivers, (1, drivers.shape[0]))
    feature.X = normalized

    plusminus = np.zeros((2, drivers.shape[1]))
    n1 = cluster1.shape[0]
    n2 = cluster2.shape[0]
    pVal = 1 - alpha
    boncorrect = pVal / (2 * dimensionality)
    q = 1 - boncorrect

    tval = t.ppf(q=q, df=n1 + n2 - 2)
    if scipy.sparse.issparse(cluster1.X):
        C1CovM = np.cov(C1Dense)
    else:
        C1CovM = np.cov(cluster1.X, rowvar=False, bias=False)
    if scipy.sparse.issparse(cluster2.X):
        C2CovM = np.cov(C2Dense)
    else:
        C2CovM = np.cov(cluster2.X, rowvar=False, bias=False)

    pooled = ((n1 - 1) * C1CovM + (n2 - 1) * C2CovM) / (n1 + n2 - 2)

    list1 = []

    for i in range(dimensionality):
        pooledVar = pooled[i, i]
        scale = tval * math.sqrt(((1 / n1) + (1 / n2)) * pooledVar)

        diff = drivers[0, i]
        plusminus[0, i] = diff - scale
        plusminus[1, i] = diff + scale
        if plusminus[0, i] < 0 and plusminus[1, i] < 0:
            # print('Gene Name: {0} Mean Difference: {1} Low: {2} High {3}'.format(cluster1.var[varName][i], diff,
            #                                                                     plusminus[0, i], plusminus[1, i]))
            gene = cluster1.var[varName][i]
            list1.append(gene)

        if plusminus[0, i] > 0 and plusminus[1, i] > 0:
            # print('Gene Name: {0} Mean Difference: {1} Low: {2} High {3}'.format(cluster1.var[varName][i], diff,
            #                                                                    plusminus[0, i], plusminus[1, i]))
            gene = cluster1.var[varName][i]
            list1.append(gene)

    standardBon = BonferroniCorrectedDifferenceMeans(cluster1, cluster2, alpha, varName, verbose=613, display=False)

    driverBon = pd.DataFrame(np.transpose(plusminus), index=cluster1.var[varName], columns=['Low', 'High'])

    filtStand = ((standardBon['High'] > 0) & (standardBon['Low'] > 0)) | (
            (standardBon['High'] < 0) & (standardBon['Low'] < 0))
    filtDriver = ((driverBon['High'] > 0) & (driverBon['Low'] > 0)) | (
            (driverBon['High'] < 0) & (driverBon['Low'] < 0))

    commonGenes = set(standardBon[filtStand].index).intersection(set(driverBon[filtDriver].index))
    intersection = standardBon.loc[list(commonGenes)]

    if display:
        if path is None:
            raise ValueError("Path Cannot Be None When Display is True")
        mapper = dict(zip(cluster1.var.index, cluster1.var['gene_short_name']))
        geneShorts = [mapper[i] for i in feature.var.index]
        feature.var.index = geneShorts

        sigs = intersection.index

        scatterWCIs = driverBon.loc[sigs]
        scatterWCIs['WRank'] = abs(scatterWCIs['Low'] + scatterWCIs['High'])
        scatterWCIs = scatterWCIs.sort_values(by='WRank', ascending=False)

        scatterBCIs = standardBon.loc[scatterWCIs.index]

        scatterBCIs['mean'] = (scatterBCIs['Low'] + scatterBCIs['High']) / 2
        scatterWCIs['mean'] = (scatterWCIs['Low'] + scatterWCIs['High']) / 2

        nonDrivers = driverBon.index.difference(sigs)

        sigPatternWeight = feature.to_df().transpose().loc[scatterWCIs.index]
        nonSigsWeights = feature.to_df().transpose().loc[nonDrivers]

        bonNonDrivers = standardBon.loc[nonDrivers]
        bonNonDrivers['mean'] = (bonNonDrivers['Low'] + bonNonDrivers['High']) / 2
        print(nonSigsWeights.shape, bonNonDrivers.shape)
        plt.scatter(bonNonDrivers['mean'], nonSigsWeights, color='black', s=pointSize)
        plt.scatter(scatterBCIs['mean'], sigPatternWeight.values, s=pointSize, color='r')

        wrap = zip(scatterBCIs['mean'].head(num_annotated), sigPatternWeight.head(num_annotated).values,
                   list(scatterBCIs.head(num_annotated).index),
                   scatterBCIs['Low'].head(num_annotated), scatterBCIs['High'].head(num_annotated))

        x1 = np.array([0, 0])
        y1 = np.array([0, 30000])

        for mean, patternWeight, txt, low, high in wrap:
            plt.annotate(txt, (mean, patternWeight + .5), fontsize=annotationSize)
            plt.plot((low, high), (patternWeight, patternWeight), '-', color='black')
        plt.plot(x1, y1, color='black')
        plt.ylim(bottom=-.25, top=max(sigPatternWeight.values)[0] + 10)
        plt.title("Latent Space " + str(featureNumber), fontsize=titleFontsize)
        plt.xlabel("Log 2 Fold Change", fontsize=axisFontsize)
        plt.ylabel("L1 Adjusted Pattern Weight", fontsize=axisFontsize)
        # Directory path and filename
        directory = os.path.dirname(path)
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path)

    return intersection, driverBon, standardBon


    # return pd.DataFrame(np.transpose(plusminus), index=cluster1.var[varName], columns=['Low', 'High'])
