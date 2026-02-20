#!/bin/bash
#SBATCH --job-name=hammon_upload
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=logs/progressive_upload_%j.out
#SBATCH --error=logs/progressive_upload_%j.err

echo "=========================================="
echo "HAMMON - Progressive Upload to S3"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Setup environment
cd /home/projects/hammon

# Run progressive upload
/home/projects/hammon/hammon-env/bin/python << 'EOF'
import boto3
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 client with higher connection pool
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

botocore_config = Config(max_pool_connections=50)
s3 = boto3.client(
    's3',
    endpoint_url='https://s3.hpccloud.lngs.infn.it/',
    aws_access_key_id='YOUR_ACCESS_KEY_HERE',
    aws_secret_access_key='YOUR_SECRET_KEY_HERE',
    config=botocore_config,
)

# Transfer config used for uploads (limits concurrency inside transfer)
transfer_config = TransferConfig(multipart_threshold=50*1024*1024, max_concurrency=4)

LOCAL_DIR = 'outputs/rectification_results'
S3_BUCKET = 'ifab.data'
S3_PREFIX = 'rectification_results'
UPLOAD_BATCH_SIZE = 20  # Upload in batches
CHECK_INTERVAL = 60  # Check every 60 seconds

# Track uploaded files
uploaded_files = set()
uploaded_file_log = 'logs/uploaded_files.txt'

# Load previously uploaded files if exists
if os.path.exists(uploaded_file_log):
    with open(uploaded_file_log, 'r') as f:
        uploaded_files = set(line.strip() for line in f)
    logger.info(f"Loaded {len(uploaded_files)} previously uploaded files")

def upload_file(local_path, s3_key):
    """Upload a single file to S3 (with simple retry), then compress heading maps locally
    only after verifying the object exists on S3.
    """
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        attempts += 1
        try:
            start = time.time()
            # Use transfer_config to control internal concurrency for multipart
            s3.upload_file(local_path, S3_BUCKET, s3_key, Config=transfer_config)
            elapsed = time.time() - start
            size_mb = os.path.getsize(local_path) / (1024 * 1024)

            # Verify object exists before mutating local file
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            except Exception as he:
                logger.warning(f"Uploaded but head_object failed for {s3_key}: {he}")
                # If head_object fails, retry upload a few times
                if attempts < max_attempts:
                    logger.info(f"Retrying upload {local_path} (attempt {attempts+1})")
                    time.sleep(2 ** attempts)
                    continue
                else:
                    return {'success': False, 'file': local_path, 'error': f'head_object failed: {he}'}

            # If it's a heading map .npy file, replace with bottom row only
            if local_path.endswith('_heading_map.npy'):
                try:
                    heading_map = np.load(local_path, allow_pickle=True)
                    original_size = os.path.getsize(local_path)

                    # Handle already-compressed 1-D arrays
                    if getattr(heading_map, 'ndim', 1) == 1:
                        logger.info(f"Skipping compression (already 1D): {os.path.basename(local_path)}")
                    else:
                        bottom_row = heading_map[-1, :]
                        np.save(local_path, bottom_row)
                        new_size = os.path.getsize(local_path)
                        saved_mb = (original_size - new_size) / (1024 * 1024)
                        logger.info(f"✅ Compressed {os.path.basename(local_path)}: "
                                   f"{original_size/1024/1024:.1f} MB → {new_size/1024:.1f} KB "
                                   f"(saved {saved_mb:.1f} MB)")
                except Exception as e:
                    logger.error(f"❌ Could not compress heading map {local_path}: {e}")

            return {
                'success': True,
                'file': local_path,
                'size_mb': size_mb,
                'time': elapsed
            }
        except Exception as e:
            logger.warning(f"Upload attempt {attempts} failed for {local_path}: {e}")
            if attempts < max_attempts:
                time.sleep(2 ** attempts)
                continue
            return {
                'success': False,
                'file': local_path,
                'error': str(e)
            }

def scan_and_upload():
    """Scan directory and upload new files"""
    # Find all files in rectification_results
    all_files = []
    for root, dirs, files in os.walk(LOCAL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            all_files.append(local_path)
    
    # Filter files not yet uploaded
    new_files = [f for f in all_files if f not in uploaded_files]
    
    if not new_files:
        return 0, 0
    
    logger.info(f"Found {len(new_files)} new files to upload")
    
    # Upload in batches with parallel workers
    uploaded_count = 0
    failed_count = 0
    total_size = 0
    
    batch_start = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for local_path in new_files[:UPLOAD_BATCH_SIZE]:
            # Calculate S3 key
            rel_path = os.path.relpath(local_path, LOCAL_DIR)
            s3_key = f"{S3_PREFIX}/{rel_path}"
            
            futures.append(executor.submit(upload_file, local_path, s3_key))
        
        for future in as_completed(futures):
            result = future.result()
            
            if result['success']:
                uploaded_count += 1
                total_size += result['size_mb']
                
                # Mark as uploaded
                uploaded_files.add(result['file'])
                
                # Save to log
                with open(uploaded_file_log, 'a') as f:
                    f.write(f"{result['file']}\n")
                
                if uploaded_count % 5 == 0:
                    logger.info(f"Uploaded {uploaded_count}/{len(futures)} files "
                               f"({total_size:.1f} MB)")
            else:
                failed_count += 1
                logger.error(f"Failed: {result['file']} - {result['error']}")
    
    batch_time = time.time() - batch_start
    if uploaded_count > 0:
        speed = total_size / batch_time if batch_time > 0 else 0
        logger.info(f"Batch complete: {uploaded_count} files, "
                   f"{total_size:.1f} MB in {batch_time:.1f}s ({speed:.2f} MB/s)")
    
    return uploaded_count, failed_count

# Main loop
logger.info("=" * 80)
logger.info("Starting progressive upload monitor")
logger.info(f"Local directory: {LOCAL_DIR}")
logger.info(f"S3 bucket: {S3_BUCKET}/{S3_PREFIX}")
logger.info(f"Check interval: {CHECK_INTERVAL}s")
logger.info("=" * 80)

total_uploaded = 0
total_failed = 0
iterations = 0

while True:
    iterations += 1
    
    # Check if rectification job is still running
    rectification_running = os.path.exists('/proc') and any(
        'slurm_rectification' in str(p) for p in Path('/proc').iterdir() if p.is_dir()
    )
    
    # Scan and upload
    uploaded, failed = scan_and_upload()
    total_uploaded += uploaded
    total_failed += failed
    
    if uploaded > 0:
        logger.info(f"Total uploaded so far: {total_uploaded} files")
    
    # If rectification is done and no new files, exit
    if not rectification_running and uploaded == 0:
        logger.info("Rectification appears complete and no new files found")
        
        # Wait one more cycle to be sure
        time.sleep(CHECK_INTERVAL)
        uploaded, failed = scan_and_upload()
        total_uploaded += uploaded
        total_failed += failed
        
        if uploaded == 0:
            logger.info("Final check: no new files. Exiting.")
            break
    
    # Wait before next check
    if uploaded == 0:
        logger.info(f"No new files. Waiting {CHECK_INTERVAL}s... "
                   f"(iteration {iterations})")
    
    time.sleep(CHECK_INTERVAL)

logger.info("=" * 80)
logger.info("PROGRESSIVE UPLOAD COMPLETED")
logger.info("=" * 80)
logger.info(f"Total uploaded: {total_uploaded}")
logger.info(f"Total failed: {total_failed}")
logger.info("=" * 80)

EOF

echo ""
echo "End time: $(date)"
echo "=========================================="
