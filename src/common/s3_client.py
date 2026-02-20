"""
Pandora Pipeline - Common Utilities
S3 Client for interacting with object storage
"""

import boto3
import geopandas as gpd
from io import BytesIO
from PIL import Image
from io import IOBase
import os
import logging

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client for reading/writing files to object storage"""
    
    def __init__(self, endpoint_url=None, access_key=None, secret_key=None):
        """
        Initialize S3 client.
        
        Args:
            endpoint_url: S3 endpoint URL (from env or config)
            access_key: AWS access key (from env or config)
            secret_key: AWS secret key (from env or config)
        """
        # Get credentials from environment if not provided
        self.endpoint_url = endpoint_url or os.getenv('S3_ENDPOINT_URL', 'https://s3.hpccloud.lngs.infn.it/')
        access_key = access_key or os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not access_key or not secret_key:
            raise ValueError("S3 credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        
        logger.info(f"S3 Client initialized with endpoint: {self.endpoint_url}")
    
    def list_buckets(self) -> list:
        """List all available buckets"""
        return self.s3.list_buckets()
    
    def list_files(self, bucket_name: str, prefix: str = '') -> list:
        """
        List files in a bucket with optional prefix.
        
        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter files
            
        Yields:
            str: File keys
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield obj['Key']
    
    def file_exists(self, bucket_name: str, file_name: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            bucket_name: Name of the bucket
            file_name: File key
            
        Returns:
            bool: True if file exists
        """
        try:
            self.s3.head_object(Bucket=bucket_name, Key=file_name)
            return True
        except:
            return False
    
    def read_file(self, bucket_name: str, file_name: str) -> bytes:
        """
        Read a file from S3 as bytes.
        
        Args:
            bucket_name: Name of the bucket
            file_name: File key
            
        Returns:
            bytes: File contents
        """
        logger.debug(f"Reading s3://{bucket_name}/{file_name}")
        obj = self.s3.get_object(Bucket=bucket_name, Key=file_name)
        return obj['Body'].read()
    
    def read_geodataframe(self, bucket_name: str, file_name: str) -> gpd.GeoDataFrame:
        """
        Read a GeoDataFrame from S3.
        
        Args:
            bucket_name: Name of the bucket
            file_name: File key
            
        Returns:
            GeoDataFrame: Loaded geodataframe
        """
        logger.info(f"Reading geodataframe from s3://{bucket_name}/{file_name}")
        data = self.read_file(bucket_name, file_name)
        
        # Try CSV first, then GeoJSON
        try:
            import pandas as pd
            return pd.read_csv(BytesIO(data))
        except:
            return gpd.read_file(BytesIO(data))
    
    def read_image(self, bucket_name: str, file_name: str) -> Image.Image:
        """
        Read an image from S3.
        
        Args:
            bucket_name: Name of the bucket
            file_name: File key
            
        Returns:
            PIL.Image: Loaded image
        """
        data = self.read_file(bucket_name, file_name)
        return Image.open(BytesIO(data))
    
    def write_object(self, data: IOBase, bucket_name: str, file_name: str) -> None:
        """
        Write a file-like object to S3.
        
        Args:
            data: File-like object (BytesIO, open file, etc.)
            bucket_name: Name of the bucket
            file_name: File key
        """
        logger.debug(f"Writing to s3://{bucket_name}/{file_name}")
        data.seek(0)  # Ensure we're at the start
        self.s3.upload_fileobj(data, bucket_name, file_name)
    
    def write_bytes(self, data: bytes, bucket_name: str, file_name: str) -> None:
        """
        Write bytes directly to S3.
        
        Args:
            data: Bytes to write
            bucket_name: Name of the bucket
            file_name: File key
        """
        logger.debug(f"Writing bytes to s3://{bucket_name}/{file_name}")
        buffer = BytesIO(data)
        self.write_object(buffer, bucket_name, file_name)
    
    def write_geodataframe(self, gdf: gpd.GeoDataFrame, bucket_name: str, file_name: str) -> None:
        """
        Write a GeoDataFrame to S3 as GeoJSON.
        
        Args:
            gdf: GeoDataFrame to write
            bucket_name: Name of the bucket
            file_name: File key
        """
        logger.info(f"Writing geodataframe to s3://{bucket_name}/{file_name}")
        buffer = BytesIO()
        gdf.to_file(buffer, driver='GeoJSON')
        self.write_object(buffer, bucket_name, file_name)
    
    def write_image(self, image: Image.Image, bucket_name: str, file_name: str, quality: int = 95) -> None:
        """
        Write an image to S3 as JPEG.
        
        Args:
            image: PIL Image to write
            bucket_name: Name of the bucket
            file_name: File key
            quality: JPEG quality (1-100)
        """
        logger.debug(f"Writing image to s3://{bucket_name}/{file_name}")
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        self.write_object(buffer, bucket_name, file_name)
    
    def download_file(self, bucket_name: str, file_name: str, local_path: str) -> None:
        """
        Download a file from S3 to local disk.
        
        Args:
            bucket_name: Name of the bucket
            file_name: File key
            local_path: Local path to save to
        """
        logger.debug(f"Downloading s3://{bucket_name}/{file_name} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(bucket_name, file_name, local_path)
    
    def upload_file(self, local_path: str, bucket_name: str, file_name: str) -> None:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Local path to upload from
            bucket_name: Name of the bucket
            file_name: File key
        """
        logger.debug(f"Uploading {local_path} to s3://{bucket_name}/{file_name}")
        self.s3.upload_file(local_path, bucket_name, file_name)
