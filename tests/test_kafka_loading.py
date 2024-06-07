from toolbox.data.loaders.kafka.kafka import KafkaLoader
from toolbox.ml_config import KafkaTopicConfiguration

# Use this to test a kafka conifugration that would normally be passed though the API

config = KafkaTopicConfiguration(**{ 
    "name": "urn_infai_ses_service_5cc05c60-5916-4b37-8991-53a93c354fe5",
    "filterType": "device_id",
    "filterValue": "urn:infai:ses:device:501f3ca8-1885-41e9-9327-8fa9c52f9528",
    "path_to_time": "value.root.lastUpdate",
    "path_to_value": "value.root.value",
    "ksql_url": "http://localhost:8088",
    "timestamp_format": "unix", #yyyy-MM-ddTHH:mm:ss.SSSZ
    "time_range_value": "13",
    "time_range_level": "h"
})

KafkaLoader(config, "exp").get_data()