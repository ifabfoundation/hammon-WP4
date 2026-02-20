#!/usr/bin/env python3
"""
Transfer files from S3 (INFN) to SharePoint Online.

This script is self-contained and stateless:
- Creates a temporary virtual environment
- Downloads files from S3 in batches
- Uploads to SharePoint using device code flow authentication (your personal credentials)
- Cleans up everything after execution (even on errors/interrupts)

Usage:
    python3 transfer_s3_to_sharepoint.py

No configuration needed - S3 credentials are embedded and SharePoint uses interactive login.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, List, Any

# Configure logging (stdout only, no file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - EMBEDDED (no environment variables needed)
# =============================================================================

class Config:
    """Configuration with embedded credentials."""
    
    # S3 Configuration (from your existing scripts)
    AWS_ACCESS_KEY_ID: str = "YOUR_ACCESS_KEY_HERE"
    AWS_SECRET_ACCESS_KEY: str = "YOUR_SECRET_KEY_HERE"
    AWS_ENDPOINT_URL: str = "https://s3.hpccloud.lngs.infn.it"
    S3_BUCKET_NAME: str = "geolander.streetview"
    S3_PREFIX: str = "2025/"
    
    # SharePoint Configuration
    SHAREPOINT_SITE_URL: str = "https://hammonit.sharepoint.com/sites/Hammon"
    SHAREPOINT_FOLDER_PATH: str = "Documenti condivisi/dati_geolander"
    
    # Azure AD - Using Microsoft's public client IDs (no admin consent required)
    # These are well-known public client IDs that work with device code flow
    AZURE_CLIENT_ID: str = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"  # Azure CLI public client
    AZURE_TENANT_ID: str = "organizations"  # Multi-tenant for work/school accounts
    
    # Transfer settings
    BATCH_SIZE: int = 5  # Files per batch (to avoid filling disk)
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate required configuration."""
        return []  # All config is embedded


# =============================================================================
# CLEANUP MANAGER
# =============================================================================

class CleanupManager:
    """Manages cleanup of temporary resources."""
    
    _instance: Optional['CleanupManager'] = None
    
    def __init__(self):
        self.temp_dir: Optional[Path] = None
        self.venv_dir: Optional[Path] = None
        self._cleanup_done: bool = False
        self._resources: List[Path] = []
    
    @classmethod
    def get_instance(cls) -> 'CleanupManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def cleanup(self) -> None:
        """Perform cleanup of all registered resources."""
        if self._cleanup_done:
            return
        
        logger.info("=" * 60)
        logger.info("STARTING CLEANUP PROCESS")
        logger.info("=" * 60)
        
        # Clean up registered resources
        for resource in self._resources:
            try:
                if resource.exists():
                    if resource.is_dir():
                        shutil.rmtree(resource, ignore_errors=True)
                        logger.info(f"✓ Removed directory: {resource}")
                    else:
                        resource.unlink()
                        logger.info(f"✓ Removed file: {resource}")
            except Exception as e:
                logger.warning(f"⚠ Failed to remove {resource}: {e}")
        
        # Clean temp directory
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"✓ Removed temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"⚠ Failed to remove temp dir: {e}")
        
        # Clean virtual environment
        if self.venv_dir and self.venv_dir.exists():
            try:
                shutil.rmtree(self.venv_dir, ignore_errors=True)
                logger.info(f"✓ Removed virtual environment: {self.venv_dir}")
            except Exception as e:
                logger.warning(f"⚠ Failed to remove venv: {e}")
        
        # Clear pip cache in temp
        pip_cache = Path(tempfile.gettempdir()) / 'pip_cache_s3_sp'
        if pip_cache.exists():
            try:
                shutil.rmtree(pip_cache, ignore_errors=True)
                logger.info(f"✓ Removed pip cache: {pip_cache}")
            except Exception as e:
                logger.warning(f"⚠ Failed to remove pip cache: {e}")
        
        self._cleanup_done = True
        logger.info("=" * 60)
        logger.info("✓ CLEANUP COMPLETED - NO TRACES LEFT")
        logger.info("=" * 60)


# Global cleanup manager
cleanup_manager = CleanupManager.get_instance()


def signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals gracefully."""
    sig_name = signal.Signals(signum).name
    logger.warning(f"\n⚠ Received {sig_name} signal. Starting graceful shutdown...")
    cleanup_manager.cleanup()
    sys.exit(130 if signum == signal.SIGINT else 143)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup on exit
atexit.register(cleanup_manager.cleanup)


@contextmanager
def managed_transfer() -> Generator[tuple[Path, Path], None, None]:
    """Context manager for the transfer operation."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = Path(tempfile.gettempdir()) / f's3_to_sharepoint_{timestamp}'
    venv_dir = Path(tempfile.gettempdir()) / f'venv_s3_sp_{timestamp}'
    
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created temp directory: {temp_dir}")
        
        cleanup_manager.temp_dir = temp_dir
        cleanup_manager.venv_dir = venv_dir
        
        yield temp_dir, venv_dir
        
    finally:
        cleanup_manager.cleanup()


# =============================================================================
# VIRTUAL ENVIRONMENT SETUP
# =============================================================================

def setup_virtual_environment(venv_dir: Path) -> Path:
    """Create and configure a temporary virtual environment."""
    logger.info("Setting up temporary virtual environment...")
    
    subprocess.run(
        [sys.executable, '-m', 'venv', str(venv_dir)],
        check=True,
        capture_output=True
    )
    logger.info(f"✓ Created virtual environment: {venv_dir}")
    
    venv_pip = venv_dir / 'bin' / 'pip'
    pip_cache = Path(tempfile.gettempdir()) / 'pip_cache_s3_sp'
    pip_cache.mkdir(exist_ok=True)
    
    # Upgrade pip silently
    subprocess.run(
        [str(venv_pip), 'install', '--upgrade', 'pip', '--cache-dir', str(pip_cache), '-q'],
        check=True,
        capture_output=True
    )
    
    # Install required packages
    packages = ['boto3', 'msal', 'requests', 'tqdm']
    
    logger.info(f"Installing packages: {', '.join(packages)}")
    result = subprocess.run(
        [str(venv_pip), 'install'] + packages + ['--cache-dir', str(pip_cache)],
        check=True,
        capture_output=True,
        text=True
    )
    logger.info("✓ Packages installed successfully")
    
    return venv_dir / 'bin' / 'python'


# =============================================================================
# TRANSFER SCRIPT (runs inside venv)
# =============================================================================

TRANSFER_SCRIPT = '''
#!/usr/bin/env python3
"""Transfer script - runs inside temporary venv."""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Set
from urllib.parse import quote

import boto3
from botocore.config import Config as BotoConfig
import msal
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration passed via environment
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_ENDPOINT_URL = os.environ['AWS_ENDPOINT_URL']
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
S3_PREFIX = os.environ['S3_PREFIX']
SHAREPOINT_SITE_URL = os.environ['SHAREPOINT_SITE_URL']
SHAREPOINT_FOLDER_PATH = os.environ['SHAREPOINT_FOLDER_PATH']
AZURE_CLIENT_ID = os.environ['AZURE_CLIENT_ID']
AZURE_TENANT_ID = os.environ['AZURE_TENANT_ID']
TEMP_DIR = Path(os.environ['TRANSFER_TEMP_DIR'])
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '5'))


class S3Client:
    """S3 client for INFN bucket."""
    
    def __init__(self):
        self.endpoint_url = AWS_ENDPOINT_URL
        self.bucket_name = S3_BUCKET_NAME
        self.prefix = S3_PREFIX
        
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=BotoConfig(
                signature_version='s3v4',
                s3={'addressing_style': 'path'}
            )
        )
        logger.info(f"✓ Connected to S3: {self.endpoint_url}/{self.bucket_name}/{self.prefix}")
    
    def list_objects(self) -> List[Dict]:
        """List all objects in the bucket with prefix."""
        objects = []
        paginator = self.client.get_paginator('list_objects_v2')
        
        logger.info(f"Listing objects in s3://{self.bucket_name}/{self.prefix}")
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if not obj['Key'].endswith('/'):
                        objects.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat()
                        })
        
        logger.info(f"Found {len(objects)} objects in S3")
        return objects
    
    def download_file(self, key: str, local_path: Path) -> bool:
        """Download a file from S3."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket_name, key, str(local_path))
            return True
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return False


class SharePointClient:
    """SharePoint client using Microsoft Graph API with device code flow."""
    
    GRAPH_URL = "https://graph.microsoft.com/v1.0"
    
    def __init__(self):
        self.site_url = SHAREPOINT_SITE_URL
        self.folder_path = SHAREPOINT_FOLDER_PATH
        self.client_id = AZURE_CLIENT_ID
        self.tenant_id = AZURE_TENANT_ID
        
        self.access_token: Optional[str] = None
        self.site_id: Optional[str] = None
        self.drive_id: Optional[str] = None
        
        self._authenticate()
        self._get_site_info()
    
    def _authenticate(self) -> None:
        """Authenticate using device code flow with your personal credentials."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("SHAREPOINT AUTHENTICATION")
        logger.info("=" * 60)
        
        app = msal.PublicClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}"
        )
        
        # Scopes for SharePoint access
        scopes = ["https://graph.microsoft.com/Files.ReadWrite.All", 
                  "https://graph.microsoft.com/Sites.ReadWrite.All"]
        
        # Try to get token from cache first
        accounts = app.get_accounts()
        if accounts:
            result = app.acquire_token_silent(scopes, account=accounts[0])
            if result and "access_token" in result:
                self.access_token = result["access_token"]
                logger.info("✓ Using cached authentication")
                return
        
        # Start device code flow
        flow = app.initiate_device_flow(scopes=scopes)
        
        if "user_code" not in flow:
            raise Exception(f"Failed to create device flow: {flow.get('error_description', 'Unknown error')}")
        
        # Display authentication instructions
        logger.info("")
        logger.info("🔐 AUTHENTICATION REQUIRED")
        logger.info("-" * 60)
        logger.info(f"1. Open this URL in a browser:")
        logger.info(f"   {flow['verification_uri']}")
        logger.info("")
        logger.info(f"2. Enter this code: {flow['user_code']}")
        logger.info("-" * 60)
        logger.info("Waiting for you to complete authentication...")
        logger.info("(Use your normal SharePoint/Microsoft 365 credentials)")
        logger.info("")
        
        # Wait for user to complete authentication
        result = app.acquire_token_by_device_flow(flow)
        
        if "access_token" in result:
            self.access_token = result["access_token"]
            logger.info("✓ Authentication successful!")
        else:
            error = result.get("error_description", result.get("error", "Unknown error"))
            raise Exception(f"Authentication failed: {error}")
    
    def _get_site_info(self) -> None:
        """Get SharePoint site and drive information."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        from urllib.parse import urlparse
        parsed = urlparse(self.site_url)
        hostname = parsed.netloc
        site_path = parsed.path.rstrip('/')
        
        # Get site info
        site_url = f"{self.GRAPH_URL}/sites/{hostname}:{site_path}"
        response = requests.get(site_url, headers=headers)
        response.raise_for_status()
        site_data = response.json()
        self.site_id = site_data['id']
        logger.info(f"✓ Found site: {site_data.get('displayName', 'Unknown')}")
        
        # Get drive info
        drives_url = f"{self.GRAPH_URL}/sites/{self.site_id}/drives"
        response = requests.get(drives_url, headers=headers)
        response.raise_for_status()
        drives_data = response.json()
        
        # Find Documents drive
        for drive in drives_data.get('value', []):
            if drive.get('name') in ['Documents', 'Documenti condivisi', 'Documenti', 'Shared Documents']:
                self.drive_id = drive['id']
                logger.info(f"✓ Found drive: {drive.get('name')}")
                break
        
        if not self.drive_id and drives_data.get('value'):
            self.drive_id = drives_data['value'][0]['id']
            logger.info(f"✓ Using drive: {drives_data['value'][0].get('name')}")
        
        if not self.drive_id:
            raise Exception("Could not find document library")
    
    def get_existing_files(self, folder_path: str) -> Set[str]:
        """Get list of existing files in SharePoint folder."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        existing = set()
        
        url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root:/{folder_path}:/children"
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 404:
                return existing
            response.raise_for_status()
            
            data = response.json()
            for item in data.get('value', []):
                if 'file' in item:  # Only files, not folders
                    existing.add(item['name'])
            
            while '@odata.nextLink' in data:
                response = requests.get(data['@odata.nextLink'], headers=headers)
                response.raise_for_status()
                data = response.json()
                for item in data.get('value', []):
                    if 'file' in item:
                        existing.add(item['name'])
                    
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 404:
                logger.warning(f"Could not list existing files: {e}")
        
        return existing
    
    def ensure_folder_exists(self, folder_path: str) -> None:
        """Ensure the target folder exists, creating if necessary."""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        parts = [p for p in folder_path.split('/') if p]
        current_path = ""
        
        for part in parts:
            if current_path:
                check_url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root:/{current_path}/{part}"
            else:
                check_url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root:/{part}"
            
            response = requests.get(check_url, headers=headers)
            
            if response.status_code == 404:
                if current_path:
                    create_url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root:/{current_path}:/children"
                else:
                    create_url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root/children"
                
                folder_data = {
                    "name": part,
                    "folder": {},
                    "@microsoft.graph.conflictBehavior": "fail"
                }
                
                create_response = requests.post(create_url, headers=headers, json=folder_data)
                if create_response.status_code not in [201, 409]:
                    create_response.raise_for_status()
                logger.info(f"✓ Created folder: {part}")
            
            current_path = f"{current_path}/{part}" if current_path else part
    
    def upload_file(self, local_path: Path, remote_folder: str, remote_name: str) -> bool:
        """Upload a file to SharePoint."""
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        file_size = local_path.stat().st_size
        max_simple_size = 4 * 1024 * 1024  # 4MB
        
        remote_path = f"{remote_folder}/{remote_name}"
        
        try:
            if file_size <= max_simple_size:
                upload_url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root:/{remote_path}:/content"
                
                with open(local_path, 'rb') as f:
                    response = requests.put(
                        upload_url,
                        headers={**headers, "Content-Type": "application/octet-stream"},
                        data=f
                    )
                response.raise_for_status()
            else:
                self._upload_large_file(local_path, remote_path, file_size, headers)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {remote_name}: {e}")
            return False
    
    def _upload_large_file(self, local_path: Path, remote_path: str, file_size: int, headers: dict) -> None:
        """Upload large file using resumable upload session."""
        session_url = f"{self.GRAPH_URL}/drives/{self.drive_id}/root:/{remote_path}:/createUploadSession"
        
        session_body = {"item": {"@microsoft.graph.conflictBehavior": "replace"}}
        
        response = requests.post(
            session_url,
            headers={**headers, "Content-Type": "application/json"},
            json=session_body
        )
        response.raise_for_status()
        
        upload_url = response.json()['uploadUrl']
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        with open(local_path, 'rb') as f:
            chunk_start = 0
            
            while chunk_start < file_size:
                chunk_end = min(chunk_start + chunk_size, file_size) - 1
                chunk_data = f.read(chunk_size)
                
                chunk_headers = {
                    "Content-Length": str(len(chunk_data)),
                    "Content-Range": f"bytes {chunk_start}-{chunk_end}/{file_size}"
                }
                
                response = requests.put(upload_url, headers=chunk_headers, data=chunk_data)
                response.raise_for_status()
                
                chunk_start = chunk_end + 1


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def main():
    """Main transfer function."""
    logger.info("=" * 60)
    logger.info("S3 TO SHAREPOINT TRANSFER")
    logger.info("=" * 60)
    
    # Initialize clients
    logger.info("")
    logger.info("Initializing S3 client...")
    s3_client = S3Client()
    
    logger.info("")
    logger.info("Initializing SharePoint client...")
    sp_client = SharePointClient()
    
    # Target folder (remove 'Documenti condivisi/' prefix if present)
    target_folder = SHAREPOINT_FOLDER_PATH
    if target_folder.startswith('Documenti condivisi/'):
        target_folder = target_folder[len('Documenti condivisi/'):]
    
    # Ensure target folder exists
    logger.info(f"Ensuring folder exists: {target_folder}")
    sp_client.ensure_folder_exists(target_folder)
    
    # Get list of files from S3
    logger.info("")
    logger.info("Getting file list from S3...")
    s3_objects = s3_client.list_objects()
    
    if not s3_objects:
        logger.info("No files found in S3. Nothing to transfer.")
        return
    
    # Get existing files in SharePoint
    logger.info("Checking existing files in SharePoint...")
    existing_files = sp_client.get_existing_files(target_folder)
    logger.info(f"Found {len(existing_files)} existing files in SharePoint")
    
    # Filter out already uploaded files
    to_transfer = []
    skipped = 0
    for obj in s3_objects:
        filename = Path(obj['key']).name
        if filename in existing_files:
            skipped += 1
        else:
            to_transfer.append(obj)
    
    logger.info(f"Files to transfer: {len(to_transfer)}")
    logger.info(f"Files skipped (already exist): {skipped}")
    
    if not to_transfer:
        logger.info("✓ All files already uploaded. Nothing to do.")
        return
    
    total_size = sum(obj['size'] for obj in to_transfer)
    logger.info(f"Total size to transfer: {format_size(total_size)}")
    
    # Process files in batches
    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING TRANSFER")
    logger.info("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    with tqdm(total=len(to_transfer), desc="Transferring", unit="file") as pbar:
        for obj in to_transfer:
            key = obj['key']
            filename = Path(key).name
            local_path = TEMP_DIR / filename
            
            try:
                # Download from S3
                pbar.set_postfix_str(f"↓ {filename[:30]}")
                if s3_client.download_file(key, local_path):
                    # Upload to SharePoint
                    pbar.set_postfix_str(f"↑ {filename[:30]}")
                    if sp_client.upload_file(local_path, target_folder, filename):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1
            finally:
                # Clean up local file immediately
                if local_path.exists():
                    local_path.unlink()
            
            pbar.update(1)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRANSFER COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Successfully transferred: {success_count}")
    if fail_count > 0:
        logger.warning(f"✗ Failed: {fail_count}")
    if skipped > 0:
        logger.info(f"⊘ Skipped (already existed): {skipped}")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\\n⚠ Transfer interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''


def run_in_venv(venv_python: Path, script_content: str, temp_dir: Path) -> int:
    """Run a script in the virtual environment."""
    script_path = temp_dir / 'transfer_script.py'
    script_path.write_text(script_content)
    
    # Pass configuration via environment
    env = {
        **os.environ,
        'PYTHONUNBUFFERED': '1',
        'AWS_ACCESS_KEY_ID': Config.AWS_ACCESS_KEY_ID,
        'AWS_SECRET_ACCESS_KEY': Config.AWS_SECRET_ACCESS_KEY,
        'AWS_ENDPOINT_URL': Config.AWS_ENDPOINT_URL,
        'S3_BUCKET_NAME': Config.S3_BUCKET_NAME,
        'S3_PREFIX': Config.S3_PREFIX,
        'SHAREPOINT_SITE_URL': Config.SHAREPOINT_SITE_URL,
        'SHAREPOINT_FOLDER_PATH': Config.SHAREPOINT_FOLDER_PATH,
        'AZURE_CLIENT_ID': Config.AZURE_CLIENT_ID,
        'AZURE_TENANT_ID': Config.AZURE_TENANT_ID,
        'TRANSFER_TEMP_DIR': str(temp_dir),
        'BATCH_SIZE': str(Config.BATCH_SIZE),
    }
    
    result = subprocess.run([str(venv_python), str(script_path)], env=env)
    return result.returncode


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> int:
    """Main entry point."""
    print()
    logger.info("=" * 60)
    logger.info("S3 TO SHAREPOINT TRANSFER - STATELESS MODE")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Source: s3://{Config.S3_BUCKET_NAME}/{Config.S3_PREFIX}")
    logger.info(f"Target: {Config.SHAREPOINT_SITE_URL}")
    logger.info(f"Folder: {Config.SHAREPOINT_FOLDER_PATH}")
    logger.info("")
    
    with managed_transfer() as (temp_dir, venv_dir):
        try:
            # Setup virtual environment
            venv_python = setup_virtual_environment(venv_dir)
            
            # Run the transfer script
            logger.info("")
            logger.info("Starting transfer process...")
            logger.info("")
            
            exit_code = run_in_venv(venv_python, TRANSFER_SCRIPT, temp_dir)
            
            if exit_code == 0:
                logger.info("")
                logger.info("✓ Transfer completed successfully!")
            else:
                logger.error(f"Transfer failed with exit code: {exit_code}")
            
            return exit_code
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Setup failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
