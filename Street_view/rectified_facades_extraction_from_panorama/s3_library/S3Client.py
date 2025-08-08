import boto3
import geopandas as gpd
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from io import IOBase
import pandas as pd

class S3Client:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url='https://s3.hpccloud.lngs.infn.it/',
            aws_access_key_id='JIUG9PGAKP2ULWP3WQ22',
            aws_secret_access_key='VLynhYvsV9JXGAqVxAMWdmo7X1THDB2p2lwxX7uy',  
        )

    def normalize_path(self, path):
        """
        Normalizza i percorsi convertendo backslash (\\) in forward slash (/)
        per compatibilità con S3.
        
        Args:
            path: Il percorso da normalizzare
            
        Returns:
            str: Il percorso normalizzato
        """
        if path is None:
            return None
        # Sostituisce tutti i backslash con forward slash
        return path.replace('\\', '/')

    def get_s3_client(self):
        return self.s3

    def create_bucket(self, bucket_name: str):
        self.s3.create_bucket(Bucket=bucket_name)

    def delete_bucket(self, bucket_name: str):
        self.s3.delete_bucket(Bucket=bucket_name)

    def list_buckets(self) -> list:
        return self.s3.list_buckets()

    def list_files(self, bucket_name: str, prefix: str) -> list:
        # Normalizza il prefisso
        prefix = self.normalize_path(prefix)
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                yield obj['Key']

    def read_file(self, bucket_name: str, file_name: str) -> bytes:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        obj = self.s3.get_object(Bucket=bucket_name, Key=file_name)
        return obj['Body'].read()

    def read_geodataframe(self, bucket_name: str, file_name: str) -> gpd.GeoDataFrame:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        return gpd.read_file(BytesIO(self.read_file(bucket_name, file_name)))

    def read_image(self, bucket_name: str, file_name: str):
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        return Image.open(BytesIO(self.read_file(bucket_name, file_name)))

    def download_file(self, bucket_name: str, file_name: str, local_path: str):
        """
        Scarica un file da S3 e lo salva in un percorso locale.
        
        Args:
            bucket_name: Nome del bucket S3
            file_name: Percorso del file su S3
            local_path: Percorso locale dove salvare il file
            
        Returns:
            str: Il percorso locale del file scaricato
        """
        # Normalizza il percorso del file S3
        file_name = self.normalize_path(file_name)
        
        # Legge il file da S3
        file_data = self.read_file(bucket_name, file_name)
        
        # Assicura che la directory di destinazione esista
        import os
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        # Scrive il file in locale
        with open(local_path, 'wb') as f:
            f.write(file_data)
            
        return local_path

    def read_shapefile(self, bucket_name: str, file_name: str) -> gpd.GeoDataFrame:
        """
        Read a complete Shapefile from S3, including all its components (.shp, .shx, .dbf, .prj, etc.)
        
        Args:
            bucket_name: Name of the S3 bucket
            file_name: Path to the main .shp file in the bucket
            
        Returns:
            GeoDataFrame containing the shapefile data
        """
        import os
        import tempfile
        import shutil
        
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        
        # Extract the base name and directory from the file path
        file_dir = os.path.dirname(file_name)
        file_basename = os.path.splitext(os.path.basename(file_name))[0]
        
        # Create a temporary directory to store the shapefile components
        temp_dir = tempfile.mkdtemp()
        
        try:
            # List all objects in the bucket with the same prefix
            paginator = self.s3.get_paginator('list_objects_v2')
            shapefile_components = []
            
            # If file_dir is empty, search at root level, otherwise include the directory
            prefix = f"{file_dir}/{file_basename}." if file_dir else f"{file_basename}."
            prefix = self.normalize_path(prefix)  # Normalizza il prefisso
            
            # Find all shapefile components
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    shapefile_components.append(key)
            
            if not shapefile_components:
                raise ValueError(f"No shapefile components found for {prefix} in bucket {bucket_name}")
            
            # Download all components to the temporary directory
            for component in shapefile_components:
                component = self.normalize_path(component)  # Normalizza il percorso del componente
                local_filename = os.path.join(temp_dir, os.path.basename(component))
                with open(local_filename, 'wb') as f:
                    self.s3.download_fileobj(bucket_name, component, f)
            
            # Read the shapefile from the temporary directory
            local_shapefile = os.path.join(temp_dir, f"{file_basename}.shp")
            gdf = gpd.read_file(local_shapefile)
            
            return gdf
        
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def write_object(self, data: IOBase, bucket_name: str, file_name: str) -> None:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        self.s3.upload_fileobj(data, bucket_name, file_name)

    def write_geodataframe(self, gdf: gpd.GeoDataFrame, bucket_name: str, file_name: str) -> None:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        buffer = BytesIO()
        gdf.to_file(buffer, driver='GeoJSON')

        buffer.seek(0)
        self.write_object(buffer, bucket_name, file_name)

    def write_image(self, image: Image.Image, bucket_name: str, file_name: str) -> None:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        buffer = BytesIO()
        image.save(buffer, format='JPEG')

        buffer.seek(0)
        self.write_object(buffer, bucket_name, file_name)

    def write_dataframe(self, df: pd.DataFrame, bucket_name: str, file_name: str, **kwargs) -> None:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        buffer = BytesIO()
        df.to_csv(buffer, **kwargs)
        
        buffer.seek(0)
        self.write_object(buffer, bucket_name, file_name)

    def write_matplotlib_figure(self, figure, bucket_name: str, file_name: str, format='png', dpi=300, **kwargs):
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        buf = BytesIO()
        figure.savefig(buf, format=format, dpi=dpi, **kwargs)

        buf.seek(0)
        self.write_object(buf, bucket_name, file_name)

    def delete_object(self, bucket_name: str, file_name: str) -> None:
        # Normalizza il percorso del file
        file_name = self.normalize_path(file_name)
        self.s3.delete_object(Bucket=bucket_name, Key=file_name)

    def delete_files_with_prefix(self, bucket_name: str, prefix: str) -> int:
        """
        Delete all files in S3 bucket with the given prefix.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: S3 key prefix to match files to delete
            
        Returns:
            int: Number of files deleted
        """
        # Normalizza il prefisso
        prefix = self.normalize_path(prefix)
        
        # List all objects with the given prefix
        paginator = self.s3.get_paginator('list_objects_v2')
        objects_to_delete = []
        
        # Collect all objects that match the prefix
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects_to_delete.append({'Key': obj['Key']})
        
        # If no objects are found, exit immediately
        if not objects_to_delete:
            print(f"No objects found with prefix '{prefix}' in bucket '{bucket_name}'")
            return 0
        
        # Delete objects in batches (up to 1000 per request)
        total_deleted = 0
        # AWS allows a maximum of 1000 objects per request
        chunk_size = 1000
        
        for i in range(0, len(objects_to_delete), chunk_size):
            chunk = objects_to_delete[i:i + chunk_size]
            response = self.s3.delete_objects(
                Bucket=bucket_name,
                Delete={
                    'Objects': chunk,
                    'Quiet': True  # Do not return the list of deleted objects
                }
            )
            total_deleted += len(chunk)
            
        print(f"Deleted {total_deleted} objects with prefix '{prefix}' from bucket '{bucket_name}'")
        return total_deleted