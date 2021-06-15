import argparse 
import pathlib 
import json 
from mascots.mtCache.mtCache import MTCache
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist


OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/exclusive_eval")
CONFIG_DIR = pathlib.Path("../mascots/mtCache/device_config")
config_array = []
for config_path in CONFIG_DIR.iterdir():
    with config_path.open("r") as f:
        config_array.append(json.load(f))


def main(rd_hist_path, bin_width):
    mt_cache = MTCache()
    output_path = OUTPUT_DIR.joinpath("{}_{}.csv".format(rd_hist_path.stem, bin_width))
    mt_cache.exhaustive_exclusive(rd_hist_path, bin_width, config_array, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cache hits for different two tier exclusive multi-tier cache.")
    parser.add_argument("rd_hist_path", type=pathlib.Path, help="Path to the RD histogram file")
    parser.add_argument("--b", default=2560, type=int, 
        help="The bin width representing the smallest unit of cache (Default: 2560 (10MB of 4KB cache blocks))")
    args = parser.parse_args()
    main(args.rd_hist_path, args.b)