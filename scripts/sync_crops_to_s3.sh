#!/bin/bash
#SBATCH --job-name=sync_crops_s3
#SBATCH --account=hammon-project
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sync_crops_s3_%j.out
#SBATCH --error=logs/sync_crops_s3_%j.err

echo "=========================================="
echo "HAMMON - Crop S3 Sync & Verification"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

cd /home/projects/hammon

/home/projects/hammon/hammon-env/bin/python << 'EOF'
import sys
sys.path.insert(0, '/home/projects/hammon/src')

import boto3
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.hpccloud.lngs.infn.it/',
    aws_access_key_id='YOUR_ACCESS_KEY_HERE',
    aws_secret_access_key='YOUR_SECRET_KEY_HERE',
)

local_dir = Path('/home/projects/hammon/outputs/crop_results')

# Get local files
local_files = sorted([f for f in local_dir.glob('*.jpg')])
logger.info(f"📂 Local crop files: {len(local_files)}")

# Get S3 file list
s3_files_dict = {}
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='ifab.data', Prefix='crop_results/')

for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('.jpg'):
                filename = key.split('/')[-1]
                s3_files_dict[filename] = {
                    'size': obj['Size'],
                    'modified': obj['LastModified']
                }

logger.info(f"☁️  S3 crop files: {len(s3_files_dict)}")

# Find missing files on S3
missing_on_s3 = []
for local_file in local_files:
    if local_file.name not in s3_files_dict:
        missing_on_s3.append(local_file)

if missing_on_s3:
    logger.warning(f"⚠️  Found {len(missing_on_s3)} files missing on S3")
    logger.info(f"📤 Uploading missing files...")
    
    uploaded = 0
    failed = 0
    
    for local_file in missing_on_s3:
        try:
            local_size = local_file.stat().st_size
            s3.upload_file(
                str(local_file),
                'ifab.data',
                f'crop_results/{local_file.name}',
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            uploaded += 1
            if uploaded % 100 == 0:
                logger.info(f"  ✓ Uploaded {uploaded}/{len(missing_on_s3)}")
        except Exception as e:
            logger.error(f"  ✗ Failed {local_file.name}: {e}")
            failed += 1
    
    logger.info(f"✅ Upload complete: {uploaded} uploaded, {failed} failed")
else:
    logger.info("✅ All local files already on S3")

# Final verification
logger.info("")
logger.info("🔍 Final verification...")

final_s3_files = set()
pages = paginator.paginate(Bucket='ifab.data', Prefix='crop_results/')
for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('.jpg'):
                filename = key.split('/')[-1]
                final_s3_files.add(filename)

final_local = set(f.name for f in local_files)

# Check consistency
if final_local == final_s3_files:
    logger.info(f"")
    logger.info(f"✅ SYNC VERIFIED")
    logger.info(f"   Total crops: {len(final_local)}")
    logger.info(f"   Local:  {len(final_local)} files")
    logger.info(f"   S3:     {len(final_s3_files)} files")
    logger.info(f"   Status: 🟢 SYNCHRONIZED")
else:
    missing_s3 = final_local - final_s3_files
    extra_s3 = final_s3_files - final_local
    
    logger.error(f"")
    logger.error(f"❌ SYNC MISMATCH")
    if missing_s3:
        logger.error(f"   Missing on S3: {len(missing_s3)} files")
    if extra_s3:
        logger.error(f"   Extra on S3: {len(extra_s3)} files")

# Save report
report_file = Path('logs/sync_crops_report.txt')
with open(report_file, 'w') as f:
    f.write(f"Crop Sync Report - {datetime.now().isoformat()}\n")
    f.write(f"=" * 60 + "\n")
    f.write(f"Local crops: {len(final_local)}\n")
    f.write(f"S3 crops: {len(final_s3_files)}\n")
    f.write(f"Status: {'✅ SYNCHRONIZED' if final_local == final_s3_files else '❌ MISMATCH'}\n")

logger.info(f"\n📝 Report saved to {report_file}")

EOF

echo ""
echo "End time: $(date)"
echo "=========================================="
