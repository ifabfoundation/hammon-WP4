#!/usr/bin/env python3
"""
Download completo di s3://geolander.streetview/2025/ in /data/hammon_data/streetview_2025/
"""
import boto3
import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Configurazione
S3_BUCKET = 'geolander.streetview'
S3_PREFIX = '2025/'
LOCAL_DIR = '/data/hammon_data/streetview_2025'
ENDPOINT_URL = 'https://s3.hpccloud.lngs.infn.it'
AWS_ACCESS_KEY = 'YOUR_ACCESS_KEY_HERE'
AWS_SECRET_KEY = 'YOUR_SECRET_KEY_HERE'

# Log file
LOG_FILE = '/home/projects/hammon/logs/s3_download.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(message):
    """Log con timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f"[{timestamp}] {message}"
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def download_from_s3():
    """Download di tutti i file da S3"""
    
    # Setup client S3
    s3_client = boto3.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    
    log("=" * 80)
    log(f"Inizio download da s3://{S3_BUCKET}/{S3_PREFIX}")
    log(f"Destinazione: {LOCAL_DIR}")
    log("=" * 80)
    
    # Contatori
    total_files = 0
    downloaded_files = 0
    skipped_files = 0
    failed_files = 0
    total_bytes = 0
    start_time = time.time()
    
    try:
        # Pagina attraverso tutti gli oggetti
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                total_files += 1
                s3_key = obj['Key']
                file_size = obj['Size']
                
                # Path locale (rimuovi il prefix S3)
                relative_path = s3_key[len(S3_PREFIX):] if s3_key.startswith(S3_PREFIX) else s3_key
                local_path = os.path.join(LOCAL_DIR, relative_path)
                
                # Salta se esiste già e ha la stessa dimensione
                if os.path.exists(local_path):
                    local_size = os.path.getsize(local_path)
                    if local_size == file_size:
                        skipped_files += 1
                        if total_files % 100 == 0:
                            log(f"Progress: {total_files} files processed, {downloaded_files} downloaded, {skipped_files} skipped")
                        continue
                
                # Crea directory se necessario
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                try:
                    # Download del file
                    s3_client.download_file(S3_BUCKET, s3_key, local_path)
                    downloaded_files += 1
                    total_bytes += file_size
                    
                    if downloaded_files % 100 == 0:
                        elapsed = time.time() - start_time
                        speed_mbps = (total_bytes / (1024*1024)) / elapsed if elapsed > 0 else 0
                        log(f"Downloaded {downloaded_files} files ({total_bytes / (1024**3):.2f} GB) - Speed: {speed_mbps:.2f} MB/s")
                    
                except Exception as e:
                    failed_files += 1
                    log(f"ERROR downloading {s3_key}: {str(e)}")
        
        # Report finale
        elapsed_time = time.time() - start_time
        log("=" * 80)
        log("DOWNLOAD COMPLETATO")
        log(f"Totale file processati: {total_files}")
        log(f"File scaricati: {downloaded_files}")
        log(f"File saltati (già esistenti): {skipped_files}")
        log(f"File falliti: {failed_files}")
        log(f"Dimensione totale scaricata: {total_bytes / (1024**3):.2f} GB")
        log(f"Tempo totale: {elapsed_time / 3600:.2f} ore")
        log(f"Velocità media: {(total_bytes / (1024*1024)) / elapsed_time:.2f} MB/s")
        log("=" * 80)
        
    except KeyboardInterrupt:
        log("\n!!! Download interrotto dall'utente !!!")
        log(f"File scaricati fino ad ora: {downloaded_files}/{total_files}")
        sys.exit(1)
    except Exception as e:
        log(f"ERRORE FATALE: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    download_from_s3()
