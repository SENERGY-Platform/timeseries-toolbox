from toolbox.data.loaders.kafka.kafka import KafkaLoader
from toolbox.data.loaders.dummy import DummyLoader
from toolbox.ml_config import Config
from toolbox.data.loaders.s3.s3 import S3DataLoader

def get_data_loader(name, data_settings):
    if name == "kafka":
        config = Config()
        experiment_name = config.EXPERIMENT_NAME
        return KafkaLoader(data_settings, experiment_name)
    elif name == "dummy":
        return DummyLoader()
    elif name == 's3':
        return S3DataLoader(data_settings.s3_url, data_settings.aws_access, data_settings.aws_secret)