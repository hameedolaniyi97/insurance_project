import pandas as pd
from zenml import pipeline 
from steps.data_loader import load_data
from steps.data_preds import encode_step, split_dataset, scale_dataset
from zenml.logger import get_logger
from steps.model_training import train_model

logger = get_logger(__name__)

#define a pipeline function 
@pipeline 
def insurace_pipeline():
    data = load_data()
    data, label_encoder = encode_step(data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    X_train, X_test, = scale_dataset(X_train,X_test)
    model = train_model(X_train, y_train, X_test, y_test)

    return model 

if __name__ =="__main__":
    insurace_pipeline()