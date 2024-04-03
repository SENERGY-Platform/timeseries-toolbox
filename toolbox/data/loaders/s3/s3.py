import boto3 
import pickle 

class S3DataLoader():   
    def __init__(
        self, 
        s3_url, 
        aws_access_key_id, 
        aws_secret_access_key
    ):
        self.s3 = boto3.resource('s3', endpoint_url=s3_url, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    def get_data(self, bucket_name, object_key):
        data = self.s3.Bucket(bucket_name).get_object(object_key)
        return pickle.loads(data)

    def put_data(self, bucket_name, file_name, data):
        self.s3.Bucket(bucket_name).put_object(Key=file_name, Body=pickle.dumps(data))

