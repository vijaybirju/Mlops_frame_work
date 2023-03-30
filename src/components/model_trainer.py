import os
import sys
sys.path.append("D:/Users/Manish/python/data science/MachinLearning/MLops")
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_objects,evaluate_models


@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split train test array')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                'Random Forest':RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'XGRegressor':XGBRegressor(),
                'CatBoosting Regression': CatBoostRegressor(allow_writing_files=False),
                'AdaBoost Regression': AdaBoostRegressor()
            }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models)
            

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            save_objects(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_scores=r2_score(y_test,predicted)
            return r2_scores
            

        except Exception as e:
            raise CustomException(e,sys)



