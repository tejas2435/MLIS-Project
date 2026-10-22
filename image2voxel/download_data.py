import os
import sys
import tarfile
import urllib.request
from tqdm import tqdm

DATA_URLS = {
    "ShapeNetRendering.tgz": "http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz",
    "ShapeNetVox32.tgz": "http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz"
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"Downloading {url} -> {output_path}")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(archive_path, extract_dir):
    print(f"Extracting {archive_path} to {extract_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print(f"Extracted successfully: {archive_path}")

def main():
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ShapeNet")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target data directory: {target_dir}")

    for filename, url in DATA_URLS.items():
        archive_path = os.path.join(target_dir, filename)
        if not os.path.exists(archive_path):
            try:
                download_file(url, archive_path)
            except Exception as e:
                print(f"Download failed for {url}: {e}")
                print(f"You can manually download {url} and place it in {target_dir}")
                continue
        else:
            print(f"Archive already downloaded: {archive_path}")

        try:
            extract_archive(archive_path, target_dir)
        except Exception as e:
            print(f"Extraction error for {archive_path}: {e}")

    print("\nDataset setup complete!")
    print(f"Check contents inside: {target_dir}")

if __name__ == "__main__":
    main()
