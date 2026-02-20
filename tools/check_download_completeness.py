#!/usr/bin/env python3
"""
Verifica completezza download confrontando locale vs S3
"""
import boto3
import os
from pathlib import Path
from collections import defaultdict

# Config
S3_BUCKET = 'geolander.streetview'
S3_PREFIX = '2025/'
LOCAL_DIR = '/data/hammon_data/streetview_2025'
ENDPOINT_URL = 'https://s3.hpccloud.lngs.infn.it'
AWS_ACCESS_KEY = 'YOUR_ACCESS_KEY_HERE'
AWS_SECRET_KEY = 'YOUR_SECRET_KEY_HERE'

print("=" * 80)
print("VERIFICA COMPLETEZZA DOWNLOAD S3")
print("=" * 80)

# Setup S3
s3_client = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Conta file su S3
print("\n[1/3] Scanning S3...")
s3_files = {}
s3_total_size = 0
s3_by_zone = defaultdict(lambda: {'count': 0, 'size': 0})

paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

for page in pages:
    if 'Contents' not in page:
        continue
    for obj in page['Contents']:
        key = obj['Key']
        size = obj['Size']
        s3_files[key] = size
        s3_total_size += size
        
        # Estrai zona
        parts = key.replace(S3_PREFIX, '').split('/')
        if parts:
            zone = parts[0]
            s3_by_zone[zone]['count'] += 1
            s3_by_zone[zone]['size'] += size

print(f"   Totale file su S3: {len(s3_files):,}")
print(f"   Dimensione totale: {s3_total_size / (1024**4):.2f} TB")

# Conta file locali
print("\n[2/3] Scanning file locali...")
local_files = {}
local_total_size = 0
local_by_zone = defaultdict(lambda: {'count': 0, 'size': 0})

for root, dirs, files in os.walk(LOCAL_DIR):
    for file in files:
        filepath = os.path.join(root, file)
        relative = os.path.relpath(filepath, LOCAL_DIR)
        s3_key = S3_PREFIX + relative
        
        try:
            size = os.path.getsize(filepath)
            local_files[s3_key] = size
            local_total_size += size
            
            # Estrai zona
            parts = relative.split('/')
            if parts:
                zone = parts[0]
                local_by_zone[zone]['count'] += 1
                local_by_zone[zone]['size'] += size
        except:
            pass

print(f"   Totale file locali: {len(local_files):,}")
print(f"   Dimensione totale: {local_total_size / (1024**4):.2f} TB")

# Confronto
print("\n[3/3] Analisi differenze...")

missing = set(s3_files.keys()) - set(local_files.keys())
extra = set(local_files.keys()) - set(s3_files.keys())
size_mismatch = []

for key in set(s3_files.keys()) & set(local_files.keys()):
    if s3_files[key] != local_files[key]:
        size_mismatch.append((key, s3_files[key], local_files[key]))

print("\n" + "=" * 80)
print("RISULTATI")
print("=" * 80)

print(f"\nFile mancanti (su S3 ma non locali): {len(missing):,}")
if missing and len(missing) <= 20:
    for f in list(missing)[:20]:
        print(f"  - {f}")
elif missing:
    print(f"  (troppi da listare, mostrare primi 20)")
    for f in list(missing)[:20]:
        print(f"  - {f}")

print(f"\nFile extra (locali ma non su S3): {len(extra):,}")
if extra and len(extra) <= 10:
    for f in list(extra)[:10]:
        print(f"  - {f}")

print(f"\nFile con dimensione diversa: {len(size_mismatch)}")
if size_mismatch:
    for key, s3_size, local_size in size_mismatch[:10]:
        print(f"  - {key}: S3={s3_size} vs Local={local_size}")

# Percentuale completamento
completeness = (len(local_files) / len(s3_files) * 100) if s3_files else 0
print(f"\n{'=' * 80}")
print(f"COMPLETAMENTO: {completeness:.2f}%")
print(f"{'=' * 80}")

# Breakdown per zona
print("\nBREAKDOWN PER ZONA:")
print(f"{'Zona':<15} {'S3 Files':>10} {'Local Files':>12} {'%':>8} {'Mancano':>10}")
print("-" * 60)

all_zones = sorted(set(list(s3_by_zone.keys()) + list(local_by_zone.keys())))
for zone in all_zones:
    s3_count = s3_by_zone[zone]['count']
    local_count = local_by_zone[zone]['count']
    pct = (local_count / s3_count * 100) if s3_count > 0 else 0
    missing = s3_count - local_count
    
    status = "✓" if pct == 100 else "⏳" if pct > 90 else "⚠"
    print(f"{zone:<15} {s3_count:>10,} {local_count:>12,} {pct:>7.1f}% {missing:>10,} {status}")

print("\n" + "=" * 80)
if missing:
    missing_size = sum(s3_files[k] for k in missing)
    print(f"Dimensione file mancanti: {missing_size / (1024**3):.2f} GB")
else:
    print("✅ DOWNLOAD COMPLETO!")
print("=" * 80)
