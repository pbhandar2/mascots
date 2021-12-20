import argparse 
import pathlib 
import json 

from mascots.mtCache.mtCache import MTCache
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist



def main(rd_hist_path, bin_width, config_array, output_dir, cost_output_dir):
    mt_cache = MTCache()
    mt_cache.exhaustive_exclusive(rd_hist_path, bin_width, config_array, output_dir, cost_output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate exclusive cache analysis data for a given workload.")
    parser.add_argument("rd_hist_path", type=pathlib.Path, 
        help="The path to the RD histogram of a workload")
    parser.add_argument("--b", type=int, default=256, 
        help="The bin width of the histogram. Default: 256 (equals 1MB if block size is 4KB)")
    parser.add_argument("--o", type=pathlib.Path, default="/research2/mtc/cp_traces/exclusive_data/4k",
        help="The directory to output the data. A new folder for the workload will be created and files generated inside.")
    parser.add_argument("--c", type=pathlib.Path, default="../mascots/mtCache/device_config",
        help="Directory containing all the device configuration files to generate data for")
    parser.add_argument("--o_c", type=pathlib.Path, default="/research2/mtc/cp_traces/exclusive_cost_data/4k",
        help="Directory to output the cost data")
    args = parser.parse_args()

    config_array = []
    for config_path in args.c.iterdir():
        with config_path.open("r") as f:
            config_array.append(json.load(f))

    # create the output directory for this workload if it doesn't already exist 
    workload_output_dir = args.o.joinpath(args.rd_hist_path.stem)
    workload_output_dir.mkdir(parents=True, exist_ok=True)

    # create the output directory for this workload if it doesn't already exist 
    workload_cost_output_dir = args.o_c.joinpath(args.rd_hist_path.stem)
    workload_cost_output_dir.mkdir(parents=True, exist_ok=True)

    main(args.rd_hist_path, args.b, config_array, workload_output_dir, workload_cost_output_dir)