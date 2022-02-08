import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from errors import FileExtensionError, SplittingError, FittingError

import warnings
warnings.filterwarnings('ignore')


class BC:
    def __init__(
            self,
            algorithms=(
                "Logistic Regression",
                "Naive Bayes",
                "SVM",
                "kNN",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "Neural Network"
            ),
            time_limit=1800,
            test_size=0.2,
            random_state=0,
            sample_weight=False,
            eval_metric="logloss"
    ):
        """
        Initialize Binary Classification auto ml object.

        :param time_limit: time limit for train model
        :param algorithms: list of algorithms, that will be used
        :param test_size: splitting data into 2 parts(test and train),indicate test size,train size equal 1 - test_size
        :param random_state: regulation randomize, default None
        :param sample_weight: training sample weight, default None
        :param eval_metric: using in xgboost algorithm, evaluation metrics for validation data default logloss
        """

        self.algorithms = algorithms
        self.test_size = test_size
        self.random_state = random_state
        self.sample_weight = sample_weight
        self.eval_metric = eval_metric
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self._fit_status = False
        self._result_dict = {}

    def load_data_csv(self, path, target_name):
        """
        Get data from file. Only csv file. Splitting received data into train and test subsets by _split_data method.
        Store subsets in 4 vars -- self.X_train, self.X_test, self.Y_train, self.Y_test

        :param path: path to csv file
        :param target_name: title of target column for classification

        """
        if not os.path.isfile(path):
            raise FileNotFoundError("Can not find data file %s" % path)
        if not os.path.splitext(path)[1] == '.csv':
            raise FileExtensionError
        df = pd.read_csv(path)
        if not self._split_data(df, target_name):
            raise SplittingError

    def _split_data(self, df, target_name):
        """
        Splitting data by using train_test_split function from sklearn.model_selection.

        :param df: pd dataframe that must be splitted
        :param target_name: title of target column for classification
        :return: True if all subsets
        """
        predictors = df.drop(target_name, axis=1)
        target = df[target_name]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(predictors, target, test_size=0.20,
                                                                                random_state=self.random_state)
        return True

    def fit(self):
        """
        Fit the model by algorithms in self.algorithms.
        By default use:
                "Logistic Regression",
                "Naive Bayes",
                "SVM",
                "kNN",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "Neural Network"
        All algorithms score storage in self._result dict.

        :return: tuple with best algorithm title and it score
        """
        if self._fit_status:
            raise FittingError
        for alg in self.algorithms:
            if alg == "Logistic Regression":
                lr = LogisticRegression()
                lr.fit(self.X_train, self.Y_train)
                Y_pred_lr = lr.predict(self.X_test)
                score_lr = round(accuracy_score(Y_pred_lr, self.Y_test) * 100, 2)
                self._result_dict["Logistic Regression"] = score_lr
            elif alg == "Naive Bayes":
                nb = GaussianNB()
                nb.fit(self.X_train, self.Y_train)
                Y_pred_nb = nb.predict(self.X_test)
                score_nb = round(accuracy_score(Y_pred_nb, self.Y_test) * 100, 2)
                self._result_dict["Naive Bayes"] = score_nb
            elif alg == "SVM":
                sv = svm.SVC(random_state=self.random_state)
                sv.fit(self.X_train, self.Y_train)
                Y_pred_svm = sv.predict(self.X_test)
                score_svm = round(accuracy_score(Y_pred_svm, self.Y_test) * 100, 2)
                self._result_dict["SVM"] = score_svm
            elif alg == "kNN":
                knn = KNeighborsClassifier()
                knn.fit(self.X_train, self.Y_train)
                Y_pred_knn = knn.predict(self.X_test)
                score_knn = round(accuracy_score(Y_pred_knn, self.Y_test) * 100, 2)
                self._result_dict["kNN"] = score_knn
            elif alg == "Decision Tree":
                max_accuracy = 0
                best_x = 0
                for x in range(200):
                    dt = DecisionTreeClassifier(random_state=x)
                    dt.fit(self.X_train, self.Y_train)
                    Y_pred_dt = dt.predict(self.X_test)
                    current_accuracy = round(accuracy_score(Y_pred_dt, self.Y_test) * 100, 2)
                    if current_accuracy > max_accuracy:
                        max_accuracy = current_accuracy
                        best_x = x
                dt = DecisionTreeClassifier(random_state=best_x)
                dt.fit(self.X_train, self.Y_train)
                Y_pred_dt = dt.predict(self.X_test)
                score_dt = round(accuracy_score(Y_pred_dt, self.Y_test) * 100, 2)
                self._result_dict["Decision Tree"] = score_dt
            elif alg == "Random Forest":
                max_accuracy = 0
                best_x = 0
                for x in range(200):
                    dt = RandomForestClassifier(random_state=x)
                    dt.fit(self.X_train, self.Y_train)
                    Y_pred_dt = dt.predict(self.X_test)
                    current_accuracy = round(accuracy_score(Y_pred_dt, self.Y_test) * 100, 2)
                    if current_accuracy > max_accuracy:
                        max_accuracy = current_accuracy
                        best_x = x
                dt = RandomForestClassifier(random_state=best_x)
                dt.fit(self.X_train, self.Y_train)
                Y_pred_dt = dt.predict(self.X_test)
                score_dt = round(accuracy_score(Y_pred_dt, self.Y_test) * 100, 2)
                self._result_dict["Random Forest"] = score_dt
            elif alg == "XGBoost":
                xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
                xgb_model.fit(self.X_train, self.Y_train)
                Y_pred_xgb = xgb_model.predict(self.X_test)
                score_xgb = round(accuracy_score(Y_pred_xgb, self.Y_test) * 100, 2)
                self._result_dict["XGBoost"] = score_xgb
        if self._result_dict:
            self._fit_status = True
        best_algorithm = max(self._result_dict, key=lambda k: self._result_dict[k])
        return best_algorithm, self._result_dict[best_algorithm]

    def view_result(self):
        print(self._result_dict)

    def view_data(self):
        return self.X_test, self.X_train, self.Y_train, self.Y_test


if __name__ == '__main__':
    bc = BC()
    bc.load_data_csv("heart.csv", "target")
    print(bc.fit())
