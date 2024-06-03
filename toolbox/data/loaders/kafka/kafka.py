import uuid
import json 
import httpx
import time 

import pandas as pd 
from ksql_query_builder import Builder, SelectContainer, CreateContainer

from toolbox.ml_config import KafkaTopicConfiguration
from toolbox.data.loaders.loader import DataLoader
from toolbox.timeseries.timestamp import todatetime 

TIME_COLUMN = "time"
VALUE_COLUMN = "value"

# EXAMPLE QUERIES
# CREATE STREAM hannes21 (device_id STRING, value STRUCT<root STRUCT<energy DOUBLE, time STRING>>) WITH (kafka_topic='urn_infai_ses_service_16e21f0a-19b4-4e5a-9247-5917ec9a4124', value_format='json', partitions=1);
# CREATE STREAM hannes22 WITH (timestamp='time', timestamp_format='yyyy-MM-dd''T''HH:mm:ss') AS SELECT device_id as device_id, value->root->time as time, value->root->energy as value FROM hannes21
# SELECT device_id, time, value FROM hannes22 WHERE device_id = 'urn:infai:ses:device:c2ac40ef-662b-4791-b0e5-a6b94f8c59ae' AND UNIX_TIMESTAMP(time) > UNIX_TIMESTAMP()-172800000.0

class KafkaLoader(DataLoader):
    def __init__(self, config: KafkaTopicConfiguration, experiment_name):
       self.topic_config = config
       self.ksql_server_url = config.ksql_url
       self.builder = Builder()
       self.stream_properties = {"ksql.streams.auto.offset.reset": "earliest"} # To query data from the beginning of the topic

    def create_unnesting_stream(self):
        # Build the `CREATE STREAM` query to access the nested value and time fields
        stream_name = str(uuid.uuid4().hex)
        create_containers = [
            CreateContainer(path=self.topic_config.path_to_time, type="STRING"), 
            CreateContainer(path=self.topic_config.path_to_value, type="DOUBLE"), 
            CreateContainer(path=self.topic_config.filterType, type="STRING")
        ]
        query = self.builder.build_create_stream_query(stream_name, self.topic_config.name, create_containers) + ";"
        print(f"create unnesting query: {query}")
        self.run_command(query)
        return stream_name

    def add_ts_format(self):
        ts_format = self.topic_config.timestamp_format.replace('T', "''T''").replace('Z', "''Z''") # KSQL requires T and Z to be escaped
        if ts_format != "unix":
            return ", timestamp_format='{ts_format}'"
        return ""

    def create_stream(self, unnesting_stream_name):
        # Create a stream that uses the time field as timestamp for further time filtering
        stream_name = str(uuid.uuid4().hex)
        select_containers = [
            SelectContainer(column_name=self.topic_config.filterType, path=self.topic_config.filterType), 
            SelectContainer(column_name=TIME_COLUMN, path=self.topic_config.path_to_time), 
            SelectContainer(column_name=VALUE_COLUMN, path=self.topic_config.path_to_value)
        ]
        select_query = self.builder.build_select_query(unnesting_stream_name, select_containers)
        query = f"CREATE STREAM {stream_name} WITH (timestamp='{TIME_COLUMN}'{self.add_ts_format()}) AS {select_query};"
        print(f"create flattened stream query: {query}")
        self.run_command(query)
        return stream_name

    def calc_unix_ts_ms(self, time_value, level):
        return round(pd.Timedelta(float(time_value), level).total_seconds() * 1000)

    def build_select_query(self, stream_name, time_value, time_level):
        # Build the `SELECT` query and filter for device ID and time range
        query = f"""SELECT {TIME_COLUMN}, {VALUE_COLUMN} FROM {stream_name}"""
        query += f" WHERE {self.topic_config.filterType} = '{self.topic_config.filterValue}'"

        unix_ts_first_point = self.calc_unix_ts_ms(time_value, time_level)
        query += f" AND UNIX_TIMESTAMP({TIME_COLUMN}) > UNIX_TIMESTAMP()-{unix_ts_first_point};"
        print(f"create select query: {query}")
        return query

    def get_data(self):
        # Creates a pd.Series with the time as index and the values as values of the Series 
        # Data is queried via KSQL from Kafka directly
        # 1. Create a stream to access the nested time and value fields
        # 2. Create a second stream that uses flat colums for time and value (this is needed as setting the time column is not possible on nested fields)
        # 3. Select from the stream
        
        unnesting_stream_name = self.create_unnesting_stream()
        stream_name = self.create_stream(unnesting_stream_name)
        time.sleep(30) # Unfortunately without this random sleep, the select query will be empty. I guess that KSQL is not ready even though the requests return successfully 
        select_query = self.build_select_query(stream_name, self.topic_config.time_range_value, self.topic_config.time_range_level)
        result = self.query_data(select_query)
        
        print(f"RETRIEVED DATA: {result[:10]}")

        self.remove_stream(stream_name)
        self.remove_stream(unnesting_stream_name)

        self.data = self.convert_result_to_series(result)
        if self.data.empty:
            raise Exception("Series is empty. Check the query.")
        
        return self.data
    
    def query_data(self, query):
        res = httpx.post(self.ksql_server_url + "/query-stream", data=json.dumps({
            "sql": query,
            "streamsProperties": self.stream_properties
        }), timeout=30, headers={'Accept': 'application/json'})
        if res.status_code != httpx.codes.OK:
            raise Exception(f"Could not query data: {res.text}")
        return res.json()

    def run_command(self, command):
        res = httpx.post(self.ksql_server_url + "/ksql", data=json.dumps({
            "ksql": command,
            "streamsProperties": self.stream_properties
        }), timeout=30, headers={'Accept': 'application/json'})
        if res.status_code != httpx.codes.OK:
            raise Exception(f"Could not run command: {res.text}")

    def remove_stream(self, stream_name):
        drop_stream_query = f'DROP STREAM {stream_name};' 
        print(f"drop query: {drop_stream_query}")
        self.run_command(drop_stream_query)
    
    def convert_result_to_series(self, result):
        data_list = result[1:] # first row contains query id and column metadata
        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[todatetime(timestamp) for timestamp, _ in data_list]).sort_index()
        return data_series