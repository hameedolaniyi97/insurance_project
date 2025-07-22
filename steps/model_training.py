from zenml import step
import pandas as pd
import numpy as np
from zenml.logger import get_logger 
from typing_extensions import Annotated
from typing import Optional, Tuple, Dict
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# confiure our logging 
logger = get_logger(__name__)

@step
def train_model(X_train: pd.DataFrame,
                y_train:pd.Series,
                X_test:pd.DataFrame,
                y_test:pd.Series) -> Annotated[Optional[RandomForestRegressor], "Model Object"]:
    model = None
    try:
        model = RandomForestRegressor(random_state=23)
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # compute the scores 
        train_rmse = root_mean_squared_error(y_train, train_preds)
        test_rmse = root_mean_squared_error(y_test, test_preds)

        logger.info(f"""C
                    Complated training the base model with metrics:
                    train model: {train_rmse}
                    test model: {test_rmse}
                    """)
        
    except Exception as err:
        logger.error(f'An error occured. Detail: {err}')

    return model
    
