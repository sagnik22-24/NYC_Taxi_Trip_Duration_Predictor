import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj, calculate_haversine_distance
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, vendor_id:int,
                     pickup_datetime:str, 
                     dropoff_datetime:str, 
                     passenger_count:int, 
                     pickup_longitude:float, 
                     pickup_latitude:float, 
                     dropoff_longitude:float, 
                     dropoff_latitude:float): 
             self.vendor_id = vendor_id
             self.pickup_datetime = pickup_datetime
             self.dropoff_datetime = dropoff_datetime
             self.passenger_count = passenger_count 
             self.pickup_longitude = pickup_longitude
             self.pickup_latitude = pickup_latitude
             self.dropoff_longitude = dropoff_longitude 
             self.dropoff_latitude = dropoff_latitude 

        def data_transform(self): 
             try: 
                  custom_data_input_dict = { 
                       'vendor_id': [self.vendor_id], 
                       'pickup_datetime': [self.pickup_datetime], 
                       'dropoff_datetime': [self.dropoff_datetime],
                       'passenger_count':[self.passenger_count],
                       'pickup_longitude':[self.pickup_longitude], 
                       'pickup_latitude': [self.pickup_latitude], 
                       'dropoff_longitude': [self.dropoff_longitude], 
                       'dropoff_latitude': [self.dropoff_latitude]
                  }

                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")

                  logging.info("Data transformation started")
                  df['distance'] = calculate_haversine_distance(df['pickup_latitude'],
                                           df['pickup_longitude'],
                                           df['dropoff_latitude'],
                                           df['dropoff_longitude'])

                  #Converting pickup_datetime and dropoff_datetime to datetime type 
                  df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], dayfirst=True) 
                  df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], dayfirst=True) 
                  
                  #day(0 = Monday, 1 = Tuesday..)
                  df['day_of_the_week'] = df['pickup_datetime'].dt.day_of_week.astype(object)  

                  #month (1 = January...6 = June)
                  df['month'] = df['pickup_datetime'].dt.month.astype(object) 

                  #hour(0 = 12am, 1 = 1am ... 23 = 11pm)                 
                  df['hour'] = df['pickup_datetime'].dt.hour.astype(object)                          

                  # Calculating duration of trip in hours
                  df['calculated_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 3600
                  
                  #New feature speed, engineered from distance and calculated_duration
                  df['speed'] = df['distance'] / df['calculated_duration']                  
                  
                  logging.info("Dataframe transformation complete")
                  return df
             
             except Exception as e:
                  logging.info("Error occured in data_transform function in prediction_pipeline")
                  raise CustomException(e,sys)       