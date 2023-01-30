import argparse
import pandas as pd
import numpy as np
from scipy.stats import mode


class KNNClassifier:

    def __init__(self, predict_list: list, k: int, X: pd.DataFrame, Y: pd.DataFrame):
        """Constructor
        :param predict_list: input values of len(4) to predict
        :param k: number of closest neighbors to base prediction from
        :param X: training dataset without target column
        :param Y: training dataset with only target column
        """
        self.predict_list = predict_list
        self.k = k
        self.X = X
        self.Y = Y

    def predict(self) -> None:
        """Prints the predicted class based on euclidean distance of nearest k-neighbors
        :return: None
        """
        distances = []
        for training_point in self.X.to_numpy():
            distances.append(euclidean_distance(np.array(self.predict_list), training_point))
        distances_df = pd.DataFrame(data=distances, columns=['distance'], index=self.Y.index)
        k_nearest_neighbors = distances_df.sort_values(by=['distance'], axis=0)[:self.k]
        neighbor_labels = self.Y.loc[k_nearest_neighbors.index]
        predicted_label = mode(neighbor_labels).mode[0]
        print('Predicted preference: %s' % predicted_label)


def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Returns the euclidean distance between two points
    :param pt1: numPy array point 1
    :param pt2: numPy array point 2
    :return: Euclidean distance between the two points
    """
    return np.sqrt(np.sum(pt1-pt2)**2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict video game preference based on given training csv')
    parser.add_argument('--train', required=True, help='csv file to train model')
    parser.add_argument('--predict', required=True, nargs=4, type=float, help='4 floating point inputs for predicting '
                                                                              'preference, '
                                                                              'height (in), weight (lbs), '
                                                                              'and gender (females are 0s '
                                                                              'and males are 1s)')
    parser.add_argument('--k', type=int, required=True, help='k-value for model')

    arg = parser.parse_args()
    # url = "https://gist.githubusercontent.com/dhar174/14177e1d874a33bfec565a07875b875a/raw/7aa9afaaacc71aa0e8bc60b38111c24e584c74d8/data.csv"
    # or put the csv file in the same directory as this file and use the following line in terminal of the pwd
    #  python knn_classifier.py --train data.csv --predict 17 160 155 1 --k 3
    dataset = pd.read_csv(arg.train)
    classifier = KNNClassifier(arg.predict, arg.k, dataset.iloc[:, :4].astype(float), dataset.iloc[:, 4])
    classifier.predict()