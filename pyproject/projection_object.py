from . import matcher
from sklearn import linear_model
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import umap


class projection:
    def __init__(self, dataset, patterns, cellTypeColumnName):
        self.dataset = dataset
        self.patterns = patterns
        overlap = matcher.getOverlap(dataset, patterns)
        self.dataset_filtered = matcher.filterSource(dataset, overlap)
        self.patterns.filtered = matcher.filterPatterns(patterns, overlap)
        self.cellTypeColumnName = cellTypeColumnName
        self.model = None
        self.pearsonMatrix = None
        self.num_cell_types = (self.dataset_filtered.obs[cellTypeColumnName].unique()).shape[0]
        self.num_patterns = patterns.X.shape[0]
        self.UMAP_COORD = None

    def non_neg_lin_reg(self, alpha, L1, iterations=10000):
        model = linear_model.ElasticNet(alpha=alpha, l1_ratio=L1, max_iter=iterations)
        model.fit(self.patterns.filtered.X.T, self.dataset_filtered.X.T)
        self.model = model
        return model

    def pearsonPlot(self, plot=True):
        color = matcher.mapCellNamesToInts(self.dataset_filtered, self.cellTypeColumnName)
        binarized = np.zeros(color.unique().shape[0], color.shape[0])
        pearson_matrix = np.empty(color.shape[0], color.unique().shape[0])
        for i in range(color.shape[0]):
            cell_type = color[i]
            binarized[cell_type][i] = 1
        for i in range(self.dataset.X.shape[0]):
            pattern = np.transpose(self.model.coef_)[:][i]
            for j in range(color.unique().shape[0]):
                cell_type = binarized[j]
                correlation = stats.pearsonr(pattern, cell_type)
                corr_np = np.corrcoef(pattern, cell_type)
                pearson_matrix[i][j] = correlation[0]
        self.pearsonMatrix = pearson_matrix
        if plot:
            plt.title("Pearson Plot", fontsize=24)
            graphic = sns.heatmap(pearson_matrix)
            plt.show()

    def UMAP_Projection(self, n_neighbors, metric='euclidean', plot=True, color="Paired", point_size=.5):
        umap_obj = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
        nd = umap_obj.fit(self.model.coef_)
        if plot:
            plt.scatter(nd[:, 0], nd[:, 1], c=[sns.color_palette(color, n_colors=12)[x] for x in color], s=point_size)
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
