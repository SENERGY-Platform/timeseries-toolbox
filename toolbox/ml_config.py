from os import environ
import uuid 
import typing 
from dataclasses import dataclass
import json

@dataclass
class KafkaTopicConfiguration:
    """"""
    name: str = None # Kafka Topic usually something like urn_infai_ses_service_be11781a-aaed-4dbb-9824-3f4e188724ec remember underscores!
    filterType: str = None # device_id, operator_id
    filterValue: str = None # Usually device id like urn:infai:ses:device:449bed81-df72-4754-a9ba-df2873046734
    path_to_time: str = None # full path to the time field, usually like value.energy.lastUpdate. Remember `value` for devices
    path_to_value: str = None
    experiment_name: str = None
    ksql_url: str = None
    timestamp_format: str = None
    time_range_value: str = None 
    time_range_level: str = None

@dataclass
class S3Configuration:
    """"""
    s3_url: str = None
    bucket_name: str = None 
    aws_secret: str = None 
    aws_access: str = None
    file_name: str = None

@dataclass
class MLSettings:
    model_name: str
    model_parameter: 'typing.Any' = None 

class Config:
    """Base config."""
    MLFLOW_URL: str = None 
    EXPERIMENT_NAME: str = None
    USER_ID: str = None 
    TASK: str = None 
    DATA_SOURCE: str = None 
    
    def __init__(self) -> None:
        self.load_from_env()
        self.parse_data_settings()
        if environ.get('TASK_SETTINGS'):
            self.parse_task_settings(environ.get('TASK_SETTINGS'))

    def load_from_env(self):
        self.MLFLOW_URL = environ['MLFLOW_URL']
        self.EXPERIMENT_NAME = environ.get('EXPERIMENT_NAME', str(uuid.uuid4().hex))
        self.USER_ID = environ['USER_ID']
        self.TASK = environ.get('TASK')
        self.DATA_SOURCE = environ['DATA_SOURCE']
        
        self.MODEL_ARTIFACT_NAME = environ.get('MODEL_ARTIFACT_NAME')
        self.METRIC_FOR_SELECTION = environ.get('METRIC_FOR_SELECTION', 'mae')
        self.METRIC_DIRECTION = environ.get('METRIC_DIRECTION', 'min')
        self.COMMIT = environ.get('COMMIT', '')
        self.MODELS = environ.get('MODELS','').split(';')
        self.PREPROCESSOR = environ.get('PREPROCESSOR')

    def parse_data_settings(self):
        if self.DATA_SOURCE == 'kafka':
            self.DATA_SETTINGS = KafkaTopicConfiguration(**json.loads(environ['DATA_SETTINGS']))
        if self.DATA_SOURCE == 's3':
            self.DATA_SETTINGS = S3Configuration(**json.loads(environ['DATA_SETTINGS']))
            
    def parse_task_settings(self, task_settings):
        task_settings = json.loads(task_settings)
        if self.TASK == 'ml_fit':
            self.TASK_SETTINGS = MLSettings(**task_settings)