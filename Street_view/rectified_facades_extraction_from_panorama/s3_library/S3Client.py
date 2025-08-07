import boto3
import geopandas as gpd
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from io import IOBase

class S3Client:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url='https://s3.hpccloud.lngs.infn.it/',
            aws_access_key_id='JIUG9PGAKP2ULWP3WQ22',
            aws_secret_access_key='VLynhYvsV9JXGAqVxAMWdmo7X1THDB2p2lwxX7uy',  
        )

    def create_bucket(self, bucket_name: str):
        self.s3.create_bucket(Bucket=bucket_name)

    def delete_bucket(self, bucket_name: str):
        self.s3.delete_bucket(Bucket=bucket_name)

    def list_buckets(self) -> list:
        return self.s3.list_buckets()

    def list_files(self, bucket_name: str) -> list:
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=''):
            for obj in page.get('Contents', []):
                yield obj['Key']

    def read_file(self, bucket_name: str, file_name: str) -> bytes:
        obj = self.s3.get_object(Bucket=bucket_name, Key=file_name)
        return obj['Body'].read()

    def read_geodataframe(self, bucket_name: str, file_name: str) -> gpd.GeoDataFrame:
        return gpd.read_file(BytesIO(self.read_file(bucket_name, file_name)))

    def read_image(self, bucket_name: str, file_name: str):
        return Image.open(BytesIO(self.read_file(bucket_name, file_name)))

    def write_object(self, data: IOBase, bucket_name: str, file_name: str) -> None:
        self.s3.upload_fileobj(data, bucket_name, file_name)

    def write_geodataframe(self, gdf: gpd.GeoDataFrame, bucket_name: str, file_name: str) -> None:
        buffer = BytesIO()
        gdf.to_file(buffer, driver='GeoJSON')

        buffer.seek(0)
        self.write_object(buffer, bucket_name, file_name)

    def write_image(self, image: Image.Image, bucket_name: str, file_name: str) -> None:
        buffer = BytesIO()
        image.save(buffer, format='JPEG')

        buffer.seek(0)
        self.write_object(buffer, bucket_name, file_name)