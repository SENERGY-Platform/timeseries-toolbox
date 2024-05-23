import uuid
import json 

from ksql import KSQLAPI
import pandas as pd 
from ksql_query_builder import Builder, SelectContainer, CreateContainer

from toolbox.ml_config import KafkaTopicConfiguration
from toolbox.data.loaders.loader import DataLoader

TIME_COLUMN = "time"
VALUE_COLUMN = "value"

class KafkaLoader(DataLoader):
    # Creates a DataFrame with columns: time, value 
    # Data is queried via KSQL from Kafka directly

    def __init__(self, config: KafkaTopicConfiguration, experiment_name):
       self.topic_config = config
       self.ksql_server_url = config.ksql_url
       self.builder = Builder()
       self.connect()

    def connect(self):
        self.client = KSQLAPI(self.ksql_server_url)
    
    def create_unnesting_stream(self):
        # Build the `CREATE STREAM` query to access the nested value and time fields
        stream_name = str(uuid.uuid4().hex)
        create_containers = [
            CreateContainer(path=self.topic_config.path_to_time, type="STRING"), 
            CreateContainer(path=self.topic_config.path_to_value, type="DOUBLE"), 
            CreateContainer(path=self.topic_config.filterType, type="STRING")
        ]
        query = self.builder.build_create_stream_query(stream_name, self.topic_config.name, create_containers)
        print(f"create unnesting query: {query}")
        self.client.ksql(query)
        return stream_name

    def create_stream(self, unnesting_stream_name):
        # Create a stream that uses the time field as timestamp for further time filtering
        stream_name = str(uuid.uuid4().hex)
        select_containers = [
            SelectContainer(column_name=self.topic_config.filterType, path=self.topic_config.filterType), 
            SelectContainer(column_name=TIME_COLUMN, path=self.topic_config.path_to_time), 
            SelectContainer(column_name=VALUE_COLUMN, path=self.topic_config.path_to_value)
        ]
        select_query = self.builder.build_select_query(unnesting_stream_name, select_containers)
        query = f"CREATE STREAM {stream_name} WITH (KAFKA_TOPIC='{unnesting_stream_name}', timestamp='{TIME_COLUMN}', timestamp_format='{self.topic_config.timestamp_format}', partitions=1, VALUE_FORMAT='json') AS {select_query}"
        print(f"create flattened stream query: {query}")
        self.client.ksql(query)
        return stream_name


    def build_select_query(self, stream_name):
        # Build the `SELECT` query and filter for device ID and time range
        query = f"""SELECT {self.topic_config.filterType}, {TIME_COLUMN}, {VALUE_COLUMN} FROM {stream_name}"""
        query += f" WHERE {self.topic_config.filterType} = '{self.topic_config.filterValue}'"
        print(f"create select query: {query}")
        return query

    def get_data(self):
        unnesting_stream_name = self.create_unnesting_stream()
        stream_name = self.create_stream(unnesting_stream_name)

        result_list = []

        try:
            select_query = self.build_select_query(stream_name)
            result = self.client.query(select_query)
            for item in result:
                result_list.append(item)    
        except Exception as e:
            print(e)
            print('Iteration done')
        
        data = self.clean_ksql_response(result_list)
        self.data = self.convert_result_to_dataframe(data)
        if self.data.empty:
            raise Exception("DataFrame is empty. Check the query.")

        drop_stream_query = f'DROP STREAM {self.stream_name}' 
        print(f"drop query: {drop_stream_query}")
        self.client.ksql(drop_stream_query)
        return self.data

    def clean_ksql_response(self, response):
        # Strip off first and last info messages
        data = []
        response = response[1:-1]
        for item in response:
            item = item.replace(",\n", "")
            item = json.loads(item)
            data.append(item)
        return data 

    def convert_result_to_dataframe(self, result):
        rows = []
        for row in result:
            values = row['row']['columns']
            time = values[0]
            value = values[1]
            rows.append({'time': time, 'value': value})
        df = pd.DataFrame(rows)
        return df
