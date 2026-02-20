#!/bin/bash
#SBATCH --job-name=hammon_download
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

echo "=========================================="
echo "HAMMON - Download Panoramas from S3"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Setup environment
cd /home/projects/hammon

# Create output directories
mkdir -p data/input/panoramas
mkdir -p logs

# Run download script with full python path
/home/projects/hammon/hammon-env/bin/python << 'EOF'
import boto3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time

# S3 client
s3 = boto3.client(
    's3',
    endpoint_url='https://s3.hpccloud.lngs.infn.it/',
    aws_access_key_id='YOUR_ACCESS_KEY_HERE',
    aws_secret_access_key='YOUR_SECRET_KEY_HERE',
)

print("📋 Loading CSV...")
df = pd.read_csv('data/input/facades_matched_summary_with_orientation.csv')

# Extract unique panorama info (FOTO format: pano_XXXXXX_YYYYYY.jpg)
print(f"📊 Total facades: {len(df)}")
print(f"📸 Unique panoramas: {df['FOTO'].nunique()}")

# Build list of panoramas to download with their S3 paths
panoramas_to_download = []
for _, row in df[['FOTO', 'RUN', 'zone']].drop_duplicates(subset=['FOTO']).iterrows():
    foto = row['FOTO']
    run = row['RUN']
    zone = row['zone']
    
    # S3 key: 2025/{zone}/pano/{RUN}/{foto}
    s3_key = f"2025/{zone}/pano/{run}/{foto}"
    
    # Local path: data/input/panoramas/{zone}/{RUN}/{foto}
    local_path = f"data/input/panoramas/{zone}/{run}/{foto}"
    
    panoramas_to_download.append({
        's3_key': s3_key,
        'local_path': local_path,
        'zone': zone,
        'run': run,
        'foto': foto
    })

total_panos = len(panoramas_to_download)
print(f"\n🎯 Panoramas to download: {total_panos}")

# Check what's already downloaded
already_downloaded = 0
to_download = []
for pano in panoramas_to_download:
    if os.path.exists(pano['local_path']):
        already_downloaded += 1
    else:
        to_download.append(pano)

print(f"✓ Already downloaded: {already_downloaded}")
print(f"📥 Need to download: {len(to_download)}")

if len(to_download) == 0:
    print("\n✅ All panoramas already downloaded!")
    exit(0)

# Download function
def download_panorama(pano_info):
    s3_key = pano_info['s3_key']
    local_path = pano_info['local_path']
    
    # Create directory
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        start = time.time()
        s3.download_file('geolander.streetview', s3_key, local_path)
        elapsed = time.time() - start
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        return {'success': True, 'foto': pano_info['foto'], 'time': elapsed, 'size': size_mb}
    except Exception as e:
        return {'success': False, 'foto': pano_info['foto'], 'error': str(e)}

# Parallel download with 10 threads
print(f"\n🚀 Starting parallel download (10 threads)...")
print("=" * 80)

start_time = time.time()
downloaded = 0
failed = 0
total_size_mb = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_panorama, pano): pano for pano in to_download}
    
    for future in as_completed(futures):
        result = future.result()
        
        if result['success']:
            downloaded += 1
            total_size_mb += result['size']
            
            # Progress update every 50 downloads
            if downloaded % 50 == 0:
                elapsed = time.time() - start_time
                speed = total_size_mb / elapsed if elapsed > 0 else 0
                eta = (len(to_download) - downloaded) * (elapsed / downloaded) if downloaded > 0 else 0
                print(f"[{downloaded}/{len(to_download)}] {result['foto']} - "
                      f"Speed: {speed:.2f} MB/s - ETA: {eta/60:.1f}m")
        else:
            failed += 1
            print(f"❌ FAILED: {result['foto']} - {result['error']}")

total_time = time.time() - start_time

print("=" * 80)
print(f"\n✅ Download completed!")
print(f"   Downloaded: {downloaded}")
print(f"   Failed: {failed}")
print(f"   Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"   Average speed: {total_size_mb/total_time:.2f} MB/s")

if failed > 0:
    print(f"\n⚠️  {failed} downloads failed. Check errors above.")
    exit(1)

EOF

echo ""
echo "End time: $(date)"
echo "=========================================="
