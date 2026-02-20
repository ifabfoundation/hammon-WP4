#!/bin/bash
#SBATCH --job-name=hammon_rect_%a
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rectification_%A_%a.out
#SBATCH --error=logs/rectification_%A_%a.err
#SBATCH --array=0-3

echo "=========================================="
echo "HAMMON - Rectification Pipeline (Batch ${SLURM_ARRAY_TASK_ID}/4)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo ""

# Setup environment
cd /home/projects/hammon

# Create output directories
mkdir -p outputs/rectification_results
mkdir -p tmp/rectification
mkdir -p logs

# Run rectification with batch splitting
/home/projects/hammon/hammon-env/bin/python << 'EOF'
import sys
sys.path.insert(0, '/home/projects/hammon/src')

import os
import pandas as pd
import yaml
import logging
import time
from pathlib import Path

# Get batch info from SLURM
BATCH_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
TOTAL_BATCHES = 4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - [BATCH {BATCH_ID}] - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print(f"🔧 Batch {BATCH_ID}/{TOTAL_BATCHES} - Importing modules...")
from rectification import PanoramaProcessor

print("✓ Modules imported successfully")

# Load config
print("📋 Loading configuration...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load CSV
print("📋 Loading facades CSV...")
df = pd.read_csv('data/input/facades_matched_summary_with_orientation.csv')

# Get unique panoramas
all_panoramas = df[['FOTO', 'RUN', 'zone']].drop_duplicates(subset=['FOTO']).to_dict('records')

# Split into batches
total_panos = len(all_panoramas)
batch_size = (total_panos + TOTAL_BATCHES - 1) // TOTAL_BATCHES
start_idx = BATCH_ID * batch_size
end_idx = min(start_idx + batch_size, total_panos)

panoramas = all_panoramas[start_idx:end_idx]
batch_total = len(panoramas)

print(f"📸 Total panoramas: {total_panos}")
print(f"📦 This batch ({BATCH_ID}): {batch_total} panoramas (index {start_idx} to {end_idx-1})")
print(f"📂 Input: data/input/panoramas/")
print(f"📂 Output: outputs/rectification_results/")
print(f"📂 Temp: tmp/rectification/batch_{BATCH_ID}/")
print("")

# Initialize processor
processor = PanoramaProcessor(
    config=config,
    temp_dir=f'tmp/rectification/batch_{BATCH_ID}',
    output_dir='outputs/rectification_results'
)

# Process panoramas
print("=" * 80)
print(f"🚀 Starting rectification for batch {BATCH_ID}...")
print("=" * 80)

start_time = time.time()
processed = 0
failed = 0
skipped = 0

for i, pano_info in enumerate(panoramas, 1):
    foto = pano_info['FOTO']
    run = pano_info['RUN']
    zone = pano_info['zone']
    
    pano_name = foto.replace('.jpg', '')
    input_path = f"data/input/panoramas/{zone}/{run}/{foto}"
    
    # Check if already processed (any output file exists)
    output_check = f"outputs/rectification_results/{pano_name}_VP_0_0.jpg"
    if os.path.exists(output_check) or os.path.exists(f"outputs/rectification_results/{pano_name}_VP_1_0.jpg"):
        skipped += 1
        if skipped % 50 == 0:
            logger.info(f"[{i}/{batch_total}] Skipped (already exists): {pano_name}")
        continue
    
    # Check if input exists
    if not os.path.exists(input_path):
        logger.warning(f"[{i}/{batch_total}] Input not found: {input_path}")
        failed += 1
        continue
    
    # Process panorama
    logger.info(f"[{i}/{batch_total}] Processing: {pano_name} (zone: {zone})")
    
    try:
        result = processor.process_single_panorama(input_path, pano_name)
        
        if result['status'] == 'success':
            processed += 1
            
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed / (elapsed / 3600) if elapsed > 0 else 0
                remaining = batch_total - processed - skipped
                eta_hours = remaining / rate if rate > 0 else 0
                
                logger.info(f"Progress: {processed}/{batch_total} | "
                           f"Rate: {rate:.1f} pano/h | "
                           f"ETA: {eta_hours:.1f}h | "
                           f"Failed: {failed} | Skipped: {skipped}")
        else:
            failed += 1
            logger.error(f"Failed: {pano_name} - {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        failed += 1
        logger.error(f"Exception processing {pano_name}: {str(e)}", exc_info=True)

total_time = time.time() - start_time

print("")
print("=" * 80)
print(f"✅ BATCH {BATCH_ID} COMPLETED")
print("=" * 80)
print(f"   Processed: {processed}")
print(f"   Skipped: {skipped}")
print(f"   Failed: {failed}")
print(f"   Total time: {total_time/3600:.2f} hours")
if total_time > 0:
    print(f"   Average rate: {processed/(total_time/3600):.1f} panoramas/hour")
print("=" * 80)

if failed > 0:
    print(f"\n⚠️  {failed} panoramas failed in batch {BATCH_ID}. Check logs for details.")

EOF

echo ""
echo "End time: $(date)"
echo "=========================================="
