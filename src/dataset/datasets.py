import gdown
import sys
from pathlib import Path
from zipfile import ZipFile

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Register datasets here
DATASETS = {
    "uci_har": {
        "url": "https://drive.google.com/uc?id=1GSCASKZxyNsEQEVwCpbqlYQjBK23oeck",
        "sentinel": DATA_DIR / "UCI HAR Dataset",
        "zip_name": "uci_har.zip",
    },
    "wisdm": {
        "url": "https://drive.google.com/uc?id=1C0VS_9J2V-7-5UoBaKgdqiUQv7R8qHYo",
        "sentinel": DATA_DIR / "WISDM",
        "zip_name": "wisdm.zip",
    }
}

# ==============================================================================
# 2. DOWNLOAD FUNCTION
# ==============================================================================

def download_and_extract(name: str, info: dict):
    sentinel = info["sentinel"]
    url = info["url"]
    zip_path = DATA_DIR / info["zip_name"]

    if sentinel.exists():
        print(f"[{name}] Dataset already found → skipping.")
        return

    print(f"[{name}] Downloading dataset...")
    gdown.download(url, str(zip_path), quiet=False)

    print(f"[{name}] Extracting...")
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"[{name}] Removing zip...")
    zip_path.unlink()

    print(f"[{name}] Done!\n")


# ==============================================================================
# 3. SELECT DATASETS TO DOWNLOAD
# ==============================================================================

def setup(datasets_to_download=None):
    """
    datasets_to_download: list of dataset names, e.g. ["uci_har", "wisdm"]
    If None → download everything
    """
    if datasets_to_download is None:
        datasets_to_download = list(DATASETS.keys())

    print("Selected datasets:", datasets_to_download, "\n")

    for ds in datasets_to_download:
        if ds not in DATASETS:
            print(f"[WARNING] Unknown dataset name: {ds}")
            continue
        download_and_extract(ds, DATASETS[ds])


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # EXAMPLES:
    # setup(["uci_har"])
    # setup(["wisdm"])
    # setup(["uci_har", "wisdm"])
    # setup()  # download ALL
    setup()

