import argparse 
import pathlib 


def main(rd_hist_path, bin_width, output_dir):
    pass 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate exclusive cache analysis data for a given workload.")
    parser.add_argument("rd_hist_path", type=pathlib.Path, 
        help="The path to the RD histogram of a workload")
    parser.add_argument("--b", type=int, default=2560, 
        help="The bin width of the histogram. Default: 2560 (equals 10MB if block size is 4KB)")
    parser.add_argument("--o", type=pathlib.Path, default="/research2/mtc/cp_traces/exclusive_eval/4k",
        help="The directory to output the data. A new folder for the workload will be created and files generated inside.")
    args = parser.parse_args()

    # create the output directory for this workload if it doesn't already exist 
    workload_output_dir = args.o.joinpath(rd_hist_path.stem)
    workload_output_dir.mkdir(parents=True, exist_ok=True)

    main(args.rd_hist_path, args.b, workload_output_dir)