import data.fetch
from config import cfg

if __name__ == "__main__":
    if not data.fetch.dataset_present(cfg["data"]["path"]):
        data.fetch.download_dataset(cfg["data"]["path"])
