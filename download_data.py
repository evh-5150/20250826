import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional

import urllib.request


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dest_dir: Path) -> Path:
    filename = url.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    if not filename:
        raise ValueError(f"Cannot infer filename from URL: {url}")
    out_path = dest_dir / filename
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    print(f"Downloading: {url} -> {out_path}")
    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.rename(out_path)
    return out_path


def extract_if_archive(path: Path, remove_archive: bool = False) -> Optional[Path]:
    suffix = ''.join(path.suffixes)
    try:
        if zipfile.is_zipfile(path):
            extract_dir = path.with_suffix("")
            print(f"Extracting ZIP: {path} -> {extract_dir}")
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(extract_dir)
            if remove_archive:
                path.unlink(missing_ok=True)
            return extract_dir
        if suffix.endswith(".tar.gz") or suffix.endswith(".tgz") or suffix.endswith(".tar"):
            extract_dir = path.with_suffix("")
            print(f"Extracting TAR: {path} -> {extract_dir}")
            with tarfile.open(path, 'r:*') as tf:
                tf.extractall(extract_dir)
            if remove_archive:
                path.unlink(missing_ok=True)
            return extract_dir
    except Exception as e:
        print(f"Warning: failed to extract {path}: {e}")
    return None


def download_http(urls: List[str], dest: Path, extract: bool, workers: int) -> None:
    ensure_dir(dest)
    def task(u: str):
        p = download_url(u, dest)
        if extract:
            extract_if_archive(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(task, urls))


def kaggle_available() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def download_kaggle(dataset: Optional[str], competition: Optional[str], dest: Path, extract: bool) -> None:
    if not kaggle_available():
        raise SystemExit("kaggle CLI not found. Install with: pip install kaggle and place kaggle.json in ~/.kaggle/")
    ensure_dir(dest)
    if dataset:
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    if competition:
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    if extract:
        for f in dest.iterdir():
            if f.is_file():
                extract_if_archive(f)


def main():
    parser = argparse.ArgumentParser(description="Download medical image datasets via HTTP or Kaggle and extract archives.")
    parser.add_argument("--dest", type=str, required=True, help="Destination directory")
    parser.add_argument("--url", type=str, action="append", help="HTTP(S) URL to download (can repeat)")
    parser.add_argument("--from_file", type=str, help="Path to a text file of URLs (one per line)")
    parser.add_argument("--kaggle_dataset", type=str, help="kaggle dataset spec, e.g., nih-chest-xrays/data")
    parser.add_argument("--kaggle_competition", type=str, help="kaggle competition name, e.g., rsna-pneumonia-detection-challenge")
    parser.add_argument("--extract", action='store_true', help="Auto-extract zip/tar archives")
    parser.add_argument("--workers", type=int, default=4, help="Parallel HTTP downloads")
    args = parser.parse_args()

    dest = Path(args.dest)
    ensure_dir(dest)

    urls: List[str] = []
    if args.url:
        urls.extend(args.url)
    if args.from_file:
        with open(args.from_file, 'r') as f:
            for line in f:
                s = line.strip()
                if s:
                    urls.append(s)

    if urls:
        download_http(urls, dest, extract=args.extract, workers=args.workers)

    if args.kaggle_dataset or args.kaggle_competition:
        download_kaggle(args.kaggle_dataset, args.kaggle_competition, dest, extract=args.extract)

    print("Done.")


if __name__ == "__main__":
    main()