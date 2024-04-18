import os 
import sys 
import pickle 
from sqlalchemy import create_engine
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging

@dataclass
class ConnectDBConfig(): 
        host = 'localhost'
        user = 'root'
        password = 'Sagnik123#'
        database = 'nyc_taxi_trips_database'
        table_name = 'nyc_taxi_trips_data'
        dataset_path:str = os.path.join('dataset', 'nyc_taxi_data.csv')

class ConnectDB():     
    def __init__(self):
         self.connect_db_config = ConnectDBConfig()  
      
    def retrieve_data(self):
        try:
            logging.info('Initiating Database Connection')
            engine = create_engine(f'mysql+mysqlconnector://{self.connect_db_config.user}:{self.connect_db_config.password}@{self.connect_db_config.host}/{self.connect_db_config.database}')
            query = f"SELECT * From {self.connect_db_config.table_name}"
            df = pd.read_sql(query, engine)
            os.makedirs(os.path.dirname(self.connect_db_config.dataset_path),exist_ok=True)
            df.to_csv(self.connect_db_config.dataset_path,index=False)
            logging.info('Copy of Dataset stored in dataset folder as a csv file')
        except Exception as e: 
            raise CustomException(e,sys)
        finally:
                engine.dispose()
                logging.info('Database connection closed')
        
def save_function(file_path, obj): 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open (file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)         

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]
# Train models
            model.fit(X_train, y_train)
# Test data
            y_test_pred = model.predict(X_test)
            #R2 Score 
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)

# Function to load a particular object 
def load_obj(file_path):
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object fuction in utils")
        raise CustomException(e,sys)

def calculate_haversine_distance(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # Average radius of the earth in km
    
    # Converting latitude and longitude from degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # Computing the differences in coordinates
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    # Haversine formula used to calculate the distance between the coordinates
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = AVG_EARTH_RADIUS * c

    return distance