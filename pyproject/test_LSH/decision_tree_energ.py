import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import time
import numpy as np
import sklearn as sk
from sklearn.metrics import accuracy_score


def main():
    # Read in file without header, header creates issues with duplicate column names
    dataframe = pd.read_csv('energMatrix_left.CSV', header=None)
    print(dataframe.head())
    print(dataframe.shape)
    # cut_gene_name is the dataframe without the first column which is the names of the genes
    cut_gene_name = dataframe.drop(dataframe.columns[0], axis=1)
    # input1 takes out the first row which is the classes and then takes the transpose for sklearn convention
    input_1 = cut_gene_name.drop(dataframe.index[0]).transpose()

    # print(outputs)
    # print(outputs.shape)
    print(cut_gene_name)
    print(str(cut_gene_name.shape) + " cut_gene_name shape")
    print(input_1)
    print(str(input_1.shape) + " input_1 shape")
    print(cut_gene_name.iloc[0])

    # create the trees balanced class_weights one gini one entropy (cost functions)
    tree_entropy = DecisionTreeClassifier(class_weight='balanced', criterion='entropy')
    print("A")
    tree_gini = DecisionTreeClassifier(class_weight='balanced', criterion='gini')
    print("B")
    in_train, in_test, out_train, out_test = train_test_split(input_1, cut_gene_name.iloc[0], test_size=.2)
    print("split")
    print(in_train)
    print(out_train)

    tree_entropy.fit(in_train, out_train)
    tree_gini.fit(in_train, out_train)

    predict_gini = tree_gini.predict(in_test)
    predict_entropy = tree_entropy.predict(in_test)
    print(str(accuracy_score(out_test, predict_entropy)) + " entropy score")
    print(str(accuracy_score(out_test, predict_gini)) + " gini score")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(time.time() - start_time)
