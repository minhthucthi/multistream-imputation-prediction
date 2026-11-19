"""
scripts/download_data.py
=========================
Script to download datasets for the MPIN project.

Supported datasets:
1. ICU (PhysioNet Challenge 2012) - Requires PhysioNet account + CITI training
2. Airquality - Public dataset (UCI or similar)

Usage:
------
    # Download ICU dataset (requires PhysioNet credentials)
    python scripts/download_data.py --dataset icu --physionet_user YOUR_USERNAME

    # Download Airquality dataset
    python scripts/download_data.py --dataset airquality

    # Download both
    python scripts/download_data.py --dataset all --physionet_user YOUR_USERNAME

    # Just check what will be downloaded (dry run)
    python scripts/download_data.py --dataset icu --dry_run

NOTES:
------
- PhysioNet ICU requires:
  1. Free account at https://physionet.org/register/
  2. Completing CITI "Data or Specimens Only Research" course
  3. Signing the DUA (Data Use Agreement) for the dataset

- After downloading, run preprocessing:
    python scripts/run_imputation.py --dataset icu --method mean  (verify data loads)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# Dataset URLs and configurations
# ===========================================================================

DATASET_INFO = {
    "icu": {
        "name": "PhysioNet Challenge 2012 (ICU)",
        "url": "https://physionet.org/files/challenge-2012/1.0.0/",
        "requires_auth": False,
        "output_dir": "data/raw/icu/",
        "description": (
            "ICU patient monitoring data: 11,988 patients, 48 hours, 37 physiological attributes.\n"
            "Missing rate: ~80%. Includes in-hospital mortality labels.\n"
            "Reference: Silva et al., 'Predicting In-hospital Mortality of ICU Patients', 2012."
        ),
        "files_to_download": [
            "set-a.tar.gz",    # Set A: 4000 patients
            "set-b.tar.gz",    # Set B: 4000 patients
            "Outcomes-a.txt",  # Labels for Set A
            "Outcomes-b.txt",  # Labels for Set B
        ]
    },
    "airquality": {
        "name": "Air Quality Multi-Station Dataset",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
        "requires_auth": False,
        "output_dir": "data/raw/airquality/",
        "description": (
            "Air quality measurements: PM2.5, PM10, SO2, NO2, CO, O3, temperature, etc.\n"
            "Multiple monitoring stations over time.\n"
            "Source: UCI Machine Learning Repository"
        ),
        "files_to_download": [
            "AirQualityUCI.zip"
        ]
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets for MPIN project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py --dataset icu --physionet_user john_doe
  python scripts/download_data.py --dataset airquality
  python scripts/download_data.py --dataset all --physionet_user john_doe
        """
    )

    parser.add_argument(
        "--dataset", type=str, default="airquality",
        choices=["icu", "airquality", "all"],
        help="Which dataset to download."
    )
    parser.add_argument(
        "--physionet_user", type=str, default=None,
        help="PhysioNet username (required for ICU dataset)."
    )
    parser.add_argument(
        "--physionet_password", type=str, default=None,
        help="PhysioNet password. If not provided, will prompt interactively."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory for downloaded data."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print download instructions without actually downloading."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist."
    )

    return parser.parse_args()


# ===========================================================================
# Download functions
# ===========================================================================

def print_instructions(dataset_key: str) -> None:
    """Print manual download instructions."""
    info = DATASET_INFO[dataset_key]
    print(f"\n{'='*60}")
    print(f"Dataset: {info['name']}")
    print(f"{'='*60}")
    print(f"Description: {info['description']}")
    print(f"\nURL: {info['url']}")
    print(f"Output directory: {info['output_dir']}")

    if info.get("requires_auth"):
        print("\n⚠️  AUTHENTICATION REQUIRED:")
        print("1. Register at: https://physionet.org/register/")
        print("2. Complete CITI course: https://www.citiprogram.org/")
        print("3. Sign DUA for challenge-2012 dataset")
        print("4. Then run with --physionet_user YOUR_USERNAME")
        print("\nManual download (browser):")
        print(f"  Visit: {info['url']}")
        print(f"  Download all files to: {info['output_dir']}")
    else:
        print("\nThis dataset does NOT require authentication.")
        print(f"\nManual download:")
        print(f"  wget {info['url']} -O {info['output_dir']}{info['files_to_download'][0]}")

    print("="*60)


def download_icu(
    username: str = None,
    password: str = None,
    output_dir: str = "data/raw/icu/",
    force: bool = False
) -> bool:
    """
    Download PhysioNet ICU Challenge 2012 dataset (Open Access).

    Parameters
    ----------
    username : str, optional
        Optional PhysioNet username.
    password : str, optional
        Optional PhysioNet password.
    output_dir : str
        Directory to save downloaded files.
    force : bool
        Re-download if files exist.

    Returns
    -------
    success : bool
    """
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://physionet.org/files/challenge-2012/1.0.0/"

    files = ["set-a.tar.gz", "set-b.tar.gz", "Outcomes-a.txt", "Outcomes-b.txt"]

    logger.info(f"Downloading PhysioNet ICU dataset to: {output_dir}")
    logger.info("This may take a moment (set-a.tar.gz & set-b.tar.gz)...")

    for file in files:
        out_path = os.path.join(output_dir, file)

        if os.path.exists(out_path) and not force:
            logger.info(f"Already exists, skipping: {file} (use --force to re-download)")
            continue

        url = base_url + file
        logger.info(f"Downloading: {url}")

        try:
            import requests
            auth = (username, password) if username and password else None
            r = requests.get(url, auth=auth, stream=True, timeout=120)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Downloaded: {file}")
        except ImportError:
            import urllib.request
            try:
                urllib.request.urlretrieve(url, out_path)
                logger.info(f"Downloaded: {file}")
            except Exception as e2:
                logger.error(f"Download failed for {file}: {e2}")
                return False
        except Exception as e:
            logger.error(f"Download failed for {file}: {e}")
            return False

    # Extract tar.gz archives
    for archive in ["set-a.tar.gz", "set-b.tar.gz"]:
        archive_path = os.path.join(output_dir, archive)
        if os.path.exists(archive_path):
            set_name = archive.replace(".tar.gz", "")
            extract_dir = os.path.join(output_dir, set_name)
            if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0 and not force:
                logger.info(f"Already extracted: {extract_dir}/")
                continue
            os.makedirs(extract_dir, exist_ok=True)
            logger.info(f"Extracting {archive}...")
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=output_dir)
            logger.info(f"Extracted to: {extract_dir}/")

    logger.info("ICU dataset download complete!")
    logger.info(f"Files saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  python scripts/run_imputation.py --dataset icu --method mean")
    return True


def download_airquality(
    output_dir: str = "data/raw/airquality/",
    force: bool = False
) -> bool:
    """
    Download Air Quality UCI dataset.

    Parameters
    ----------
    output_dir : str
    force : bool

    Returns
    -------
    success : bool
    """
    os.makedirs(output_dir, exist_ok=True)

    # UCI Air Quality Dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    out_path = os.path.join(output_dir, "AirQualityUCI.zip")

    if os.path.exists(out_path) and not force:
        logger.info(f"AirQualityUCI.zip already exists. Use --force to re-download.")
    else:
        logger.info(f"Downloading Air Quality dataset from UCI...")
        logger.info(f"URL: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, out_path)
            logger.info(f"Downloaded: AirQualityUCI.zip")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("\nManual download alternative:")
            logger.info(f"  curl -o {out_path} {url}")
            logger.info(f"  OR: wget -O {out_path} {url}")
            return False

    # Extract
    import zipfile
    logger.info("Extracting AirQualityUCI.zip...")
    try:
        with zipfile.ZipFile(out_path, "r") as zf:
            zf.extractall(output_dir)
        logger.info(f"Extracted to: {output_dir}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False

    logger.info("Airquality dataset download complete!")
    logger.info(f"Files saved to: {output_dir}")

    # TODO: Convert Excel/CSV format to expected format
    logger.info(
        "\nNote: AirQualityUCI is in Excel format (.xlsx). "
        "Run preprocessing to convert:\n"
        "  python scripts/run_imputation.py --dataset airquality --method mean"
    )
    return True


def check_existing_data(output_dir: str, dataset_key: str) -> bool:
    """Check if dataset already exists."""
    if not os.path.exists(output_dir):
        return False
    files = os.listdir(output_dir)
    return len(files) > 0


def main():
    args = parse_args()

    datasets_to_download = (
        ["icu", "airquality"] if args.dataset == "all"
        else [args.dataset]
    )

    logger.info("MPIN Project – Dataset Download Script")
    logger.info("=" * 50)

    for dataset_key in datasets_to_download:
        info = DATASET_INFO[dataset_key]
        output_dir = args.output_dir or info["output_dir"]

        if args.dry_run:
            print_instructions(dataset_key)
            continue

        # Check if already downloaded
        if check_existing_data(output_dir, dataset_key) and not args.force:
            logger.info(
                f"Data already exists in {output_dir}. "
                "Use --force to re-download."
            )
            continue

        if dataset_key == "icu":
            password = args.physionet_password
            if args.physionet_user and not password:
                import getpass
                password = getpass.getpass(f"PhysioNet password for '{args.physionet_user}': ")

            success = download_icu(
                username=args.physionet_user,
                password=password,
                output_dir=output_dir,
                force=args.force
            )
            if not success:
                logger.error("ICU download failed. See instructions above.")

        elif dataset_key == "airquality":
            success = download_airquality(output_dir=output_dir, force=args.force)
            if not success:
                logger.error("Airquality download failed.")
                print_instructions("airquality")

    logger.info("\nDone!")
    logger.info("Next step:")
    logger.info("  python notebooks/01_data_exploration.ipynb  (Jupyter)")
    logger.info("  python scripts/run_imputation.py --dataset icu --method all")


if __name__ == "__main__":
    main()
