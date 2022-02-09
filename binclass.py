import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb

from typing import Union, Tuple
from pprint import pprint

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from errors import FileExtensionError, SplittingError, FittingError, MetricError, PredictionError

warnings.filterwarnings('ignore')


class BC:
    def __init__(
            self,
            algorithms: Union[tuple, list] =
            (
                "Logistic Regression",
                "Naive Bayes",
                "SVM",
                "kNN",
                "Decision Tree",
                "Random Forest",
                "XGBoost"
            ),
            metric: str = "accuracy",
            test_size: float = 0.2,
            random_state: int = 0,
            sample_weight: bool = False,
            eval_metric: str = "logloss"
    ):
        """
        Initialize Binary Classification auto ml object.

        :param metric:
        :param algorithms: list of algorithms, that will be used
        :param test_size: splitting data into 2 parts(test and train),indicate test size,train size equal 1 - test_size
        :param random_state: regulation randomize, default None
        :param sample_weight: training sample weight, default None
        :param eval_metric: using in xgboost algorithm, evaluation metrics for validation data default logloss
        """

        self.algorithms = algorithms
        self.metric = metric
        self.test_size = test_size
        self.random_state = random_state
        self.sample_weight = sample_weight
        self.eval_metric = eval_metric
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.train_algo = None
        self._metric_dict = {
            "accuracy": accuracy_score,
            "precision": average_precision_score,
            "f1_score": f1_score,
        }
        self._algo_dict = {
            "Logistic Regression": LogisticRegression,
            "Naive Bayes": GaussianNB,
            "SVM": svm,
            "kNN": KNeighborsClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "XGBoost": xgb
        }
        self._classificator = None
        self._fit_status = False
        self._result_dict = {m: {} for m in self._metric_dict.keys()}

    def load_data_csv(self, path: str, target_name: str) -> None:
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

    def _split_data(self, df: pd.DataFrame, target_name: str) -> bool:
        """
        Splitting data by using train_test_split function from sklearn.model_selection.

        :param df: pd dataframe that must be splitted
        :param target_name: title of target column for classification
        :return: True if all subsets are created
        """
        predictors = df.drop(target_name, axis=1)
        target = df[target_name]
        try:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(predictors, target, test_size=0.20,
                                                                                    random_state=self.random_state)
        except Exception as e:
            raise e
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
                "XGBoost"
        All algorithms metrics score storage in self._result dict.

        :return: tuple with best algorithm title and it score
        """
        if self._fit_status:
            raise FittingError
        if self.metric not in self._metric_dict:
            raise MetricError
        for alg in self.algorithms:

            if alg == "Logistic Regression":
                print("Fitting by", alg)
                lr = LogisticRegression()
                lr.fit(self.X_train, self.Y_train)
                Y_pred_lr = lr.predict(self.X_test)
                self._get_metric(Y_pred_lr, alg)

            elif alg == "Naive Bayes":
                nb = GaussianNB()
                nb.fit(self.X_train, self.Y_train)
                Y_pred_nb = nb.predict(self.X_test)
                self._get_metric(Y_pred_nb, alg)
                print("Fitting by", alg)

            elif alg == "SVM":
                print("Fitting by", alg)
                sv = svm.SVC(random_state=self.random_state)
                sv.fit(self.X_train, self.Y_train)
                Y_pred_svm = sv.predict(self.X_test)
                self._get_metric(Y_pred_svm, alg)

            elif alg == "kNN":
                print("Fitting by", alg)
                knn = KNeighborsClassifier()
                knn.fit(self.X_train, self.Y_train)
                Y_pred_knn = knn.predict(self.X_test)
                self._get_metric(Y_pred_knn, alg)

            elif alg == "Decision Tree":
                print("Fitting by", alg)
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
                self._get_metric(Y_pred_dt, alg)

            elif alg == "Random Forest":
                print("Fitting by", alg)
                max_accuracy = 0
                best_x = 0
                for x in range(200):
                    rf = RandomForestClassifier(random_state=x)
                    rf.fit(self.X_train, self.Y_train)
                    Y_pred_rf = rf.predict(self.X_test)
                    current_accuracy = round(accuracy_score(Y_pred_rf, self.Y_test) * 100, 2)
                    if current_accuracy > max_accuracy:
                        max_accuracy = current_accuracy
                        best_x = x
                rf = RandomForestClassifier(random_state=best_x)
                rf.fit(self.X_train, self.Y_train)
                Y_pred_rf = rf.predict(self.X_test)
                self._get_metric(Y_pred_rf, alg)

            elif alg == "XGBoost":
                print("Fitting by", alg)
                xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
                xgb_model.fit(self.X_train, self.Y_train)
                Y_pred_xgb = xgb_model.predict(self.X_test)
                self._get_metric(Y_pred_xgb, alg)

            else:
                raise FittingError("Wrong algorithm")

        best_algorithm, result = self._best_algo()

        if best_algorithm:
            self._fit_status = True
            self.train_algo = best_algorithm
            if best_algorithm in ["Decision Tree", "Random Forest"]:
                if best_algorithm == "Decision Tree":
                    self._classificator = dt
                else:
                    self._classificator = rf
        print("Best algorithm is:", best_algorithm, "score is:", result)
        return best_algorithm, result

    def _get_metric(self, Y_pred: np.ndarray, alg: str) -> None:
        """
        Fill tne result metric dictionary by scores of algorithms
        :param Y_pred: predicted values
        :param alg: name of the algorithm
        """
        for m, f in self._metric_dict.items():
            self._result_dict[m][alg] = round(f(Y_pred, self.Y_test) * 100, 2)

    def _best_algo(self) -> Tuple[str, float]:
        """
        Get the best algorithms for selected metric.
        By default metric is accuracy
        :return: name of algorithm, algorithms score by metric
        """
        best_algorithm = max(self._result_dict[self.metric], key=lambda k: self._result_dict[self.metric][k])

        return best_algorithm, self._result_dict[self.metric][best_algorithm]

    def predict(self, dt: pd.DataFrame) -> np.ndarray:
        """
        Predict by fitted model
        :param dt: dataframe with data
        :return: predicition values
        """
        if not self._fit_status:
            raise FittingError("This model hasn't been fitted")
        if not isinstance(dt, pd.DataFrame):
            raise TypeError("Wrong type of data on prediction. Only pandas dataframe is allowed")
        try:
            algorithm_func = self._algo_dict[self.train_algo]
            if algorithm_func.__name__ in ["RandomForestClassifier", "DecisionTreeClassifier"]:
                pred = self._classificator.predict(dt)
            else:
                pred = algorithm_func.predict(self.X_test)
        except Exception:
            raise PredictionError
        print("Prediction is:", pred)
        return pred

    def view_result(self):
        """
        Method for viewing result metric dictionary

        """
        pprint(self._result_dict)

    def view_data(self):
        """
        Method for viewing fitting data
        :return X_test, X_train, Y_train, Y_test
        """
        return self.X_test, self.X_train, self.Y_train, self.Y_test
