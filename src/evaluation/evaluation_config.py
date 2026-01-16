import pandas as pd
from collections.abc import Callable
class EvaluationConfig:
    """
    Contains the basic information to run an evaluation on a dataset
    """
    def __init__(self,
                 function: Callable,
                 function_name: str,
                 model_name: str,
                 dataset: pd.DataFrame,
                 dataset_name: str,
                 description: str,
                 use_translation: bool=False,
                 save_predictions: bool=False):
        self.function = function
        self.function_name = function_name
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.description = description
        self.use_translation = use_translation#
        self.save_predictions = save_predictions
