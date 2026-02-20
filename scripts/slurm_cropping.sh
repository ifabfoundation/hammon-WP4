#!/bin/bash
#SBATCH --job-name=hammon_cropping
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cropping_%j.out
#SBATCH --error=logs/cropping_%j.err

echo "=========================================="
echo "HAMMON - Cropping Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Setup environment
cd /home/projects/hammon

# Create output directories
mkdir -p outputs/crop_results
mkdir -p tmp/cropping
mkdir -p logs

# Run cropping
/home/projects/hammon/hammon-env/bin/python << 'EOF'
import sys
sys.path.insert(0, '/home/projects/hammon/src')

import os
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("🔧 Importing cropping modules...")
import yaml
import time
import boto3
from concurrent.futures import ThreadPoolExecutor
from cropping import BuildingExtractor

print("📋 Loading configuration...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize S3 client for background upload
s3 = boto3.client(
    's3',
    endpoint_url=config['s3']['endpoint_url'],
    aws_access_key_id='YOUR_ACCESS_KEY_HERE',
    aws_secret_access_key='YOUR_SECRET_KEY_HERE',
)

def upload_to_s3_background(file_path: Path):
    """Upload a file to S3 in background."""
    try:
        s3_key = f"crop_results/{file_path.name}"
        s3.upload_file(
            str(file_path),
            'ifab.data',
            s3_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        logger.debug(f"✓ Uploaded to S3: {file_path.name}")
        return True
    except Exception as e:
        logger.warning(f"S3 upload failed for {file_path.name}: {e}")
        return False

# Create upload thread pool
upload_executor = ThreadPoolExecutor(max_workers=2)
upload_futures = []

print("📋 Loading CSV...")
df = pd.read_csv('data/input/facades_extended_local.csv')

print(f"📊 Total facades to crop: {len(df)}")
print(f"📂 Input: outputs/rectification_results/")
print(f"📂 Output: outputs/crop_results/")
print("")

# Initialize extractor
extractor = BuildingExtractor(
    config=config,
    temp_dir='tmp/cropping',
    output_dir='outputs/crop_results'
)

# Check which panoramas have been rectified
rectified_dir = Path('outputs/rectification_results')
available_panos = set()
for item in rectified_dir.glob('pano_*'):
    # Extract pano name from any file (e.g., pano_000047_000130_VP_1_0.jpg)
    pano_name = '_'.join(item.stem.split('_')[:3])  # pano_000047_000130
    available_panos.add(pano_name)

print(f"✓ Found {len(available_panos)} rectified panoramas")

# Filter facades that have rectified panoramas available
# FOTO column already contains pano_XXXXXX_YYYYYY.jpg format
df['pano_name'] = df['FOTO'].str.replace('.jpg', '', regex=False)
df_available = df[df['pano_name'].isin(available_panos)].copy()

print(f"✓ {len(df_available)} facades can be cropped from available panoramas")
print("")

# Process facades
start_time = time.time()
processed = 0
failed = 0
skipped = 0

for idx, row in df_available.iterrows():
    foto = row['FOTO']
    pano_name = row['pano_name']
    
    # Check if already processed
    output_file = Path(f"outputs/crop_results/final_{pano_name}_final.jpg")
    if output_file.exists():
        skipped += 1
        if skipped % 50 == 0:
            logger.info(f"[{idx+1}/{len(df_available)}] Skipped (already exists): {pano_name}")
        continue
    
    logger.info(f"[{idx+1}/{len(df_available)}] Processing facade: {pano_name}")
    
    try:
        result = extractor.process_single_building(
            row=row,
            rectified_image_folder='outputs/rectification_results',
            save_final=True
        )
        
        if result:
            processed += 1
            
            # Upload to S3 in background
            result_path = Path(result)
            if result_path.exists():
                future = upload_executor.submit(upload_to_s3_background, result_path)
                upload_futures.append(future)
            
            # Progress report
            if processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / (elapsed / 3600)  # facades/hour
                remaining = len(df_available) - processed - skipped
                eta_hours = remaining / rate if rate > 0 else 0
                
                # Count successful uploads
                uploads_done = sum(1 for f in upload_futures if f.done() and f.result())
                
                logger.info(f"Progress: {processed}/{len(df_available)} | "
                           f"Rate: {rate:.1f} facade/h | "
                           f"ETA: {eta_hours:.1f}h | "
                           f"Failed: {failed} | Skipped: {skipped} | "
                           f"Uploaded: {uploads_done}")
        else:
            failed += 1
            logger.warning(f"Failed to process facade: {pano_name}")
            
    except Exception as e:
        failed += 1
        logger.error(f"Exception processing {pano_name}: {str(e)}", exc_info=True)

total_time = time.time() - start_time

# Wait for all uploads to complete
print("")
print("⏳ Waiting for remaining S3 uploads to complete...")
upload_executor.shutdown(wait=True)

# Count successful uploads
uploads_success = sum(1 for f in upload_futures if f.result())
uploads_failed = len(upload_futures) - uploads_success

print("")
print("=" * 80)
print("✅ CROPPING COMPLETED")
print("=" * 80)
print(f"   Processed: {processed}")
print(f"   Skipped: {skipped}")
print(f"   Failed: {failed}")
print(f"   Total time: {total_time/3600:.2f} hours")
print(f"   Average rate: {processed/(total_time/3600):.1f} facades/hour")
print("")
print("☁️  S3 UPLOAD STATUS")
print(f"   Uploaded: {uploads_success}")
print(f"   Failed: {uploads_failed}")
print("=" * 80)

if failed > 0:
    print(f"\n⚠️  {failed} facades failed. Check logs for details.")
if uploads_failed > 0:
    print(f"\n⚠️  {uploads_failed} S3 uploads failed. Check logs for details.")

EOF

echo ""
echo "End time: $(date)"
echo "=========================================="
