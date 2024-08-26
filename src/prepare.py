import argparse

import data.fetch
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--root_dir", type=str,
                        help='Root directory where to download the dataset', default=cfg["data"]["path"], required=False)
    parser.add_argument("-s", "--split",
                        help='Whether to split (maybe again) the dataset, based on the config data.', default=False, required=False, action="store_true")

    args = parser.parse_args()

    if not data.fetch.dataset_present(args.root_dir):
        data.fetch.download_dataset(args.root_dir)

    if args.split:
        data.fetch.split_chunk(f'{args.root_dir}/Chunk_1')
