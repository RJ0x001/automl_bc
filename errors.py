class FileExtensionError(Exception):
    def __init__(self, *args, **kwargs):
        default_message = "Wrong file extension. Expectation .csv"
        if not args:
            args = (default_message,)
        super().__init__(*args, **kwargs)


class SplittingError(Exception):
    def __init__(self, *args, **kwargs):
        default_message = "Error with splitting data by train_test_split. One subset is empty"
        if not args:
            args = (default_message,)
        super().__init__(*args, **kwargs)


class FittingError(Exception):
    def __init__(self, *args, **kwargs):
        default_message = "This model has already been fitted"
        if not args:
            args = (default_message,)
        super().__init__(*args, **kwargs)


class MetricError(Exception):
    def __init__(self, *args, **kwargs):
        default_message = "Can't find this metric. You can use one of them (accuracy, precision, f1_score)"
        if not args:
            args = (default_message,)
        super().__init__(*args, **kwargs)


class PredictionError(Exception):
    def __init__(self, *args, **kwargs):
        default_message = "Error with prediction by already fitting model"
        if not args:
            args = (default_message,)
        super().__init__(*args, **kwargs)
