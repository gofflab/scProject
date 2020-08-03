import sklearn as sk
import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd
from sklearn.model_selection import train_test_split


# this will return the columns in the dataset that have the feature we are looking for
class Training_Dataset:
    def __init__(self, Pattern_Matrix, A_Matrix, Dataset, column_in_A, lower_bound, upper_bound):
        self.pattern_matrix = sc.read(Pattern_Matrix)
        self.A_Matrix = sc.read(A_Matrix)
        self.dataset = sc.read(Dataset)
        self.column_in_A = column_in_A
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def feature_vectors_positive(self):
        indices = np.where(self.pattern_matrix.X[self.column_in_A][:] > self.lower_bound)
        training_samples = self.dataset.X[:, indices]
        return training_samples

    def feature_vectors_negative(self):
        indices = np.where(self.pattern_matrix.X[self.column_in_A][:] < self.upper_bound)
        training_samples = self.dataset.X[:, indices]
        return training_samples

    def construct_training_set(self):
        positive_training_data = self.feature_vectors_positive()
        negative_training_data = self.feature_vectors_negative()
        matrix = np.concatenate(positive_training_data, negative_training_data, axis=1)
        true = np.ones(positive_training_data.shape[1])
        false = np.zeros(negative_training_data.shape[1])
        classes = pd.DataFrame(np.concatenate(true, false, axis=1))
        return ad.AnnData(matrix, var=classes)

    def decision_tree(self, verbose=True):
        tree = sk.tree.DecisionTreeClassifier(class_weight="balanced", criterion='entropy')
        anndata = self.construct_training_set()
        input_train, input_test, output_train, output_test = train_test_split(anndata.X, anndata.var, test_size=.2)
        tree.fit(input_train, output_train)
        prediction = tree.predict(input_test)
        print("accuracy ", sk.metrics.accuracy_score(output_test, prediction))
