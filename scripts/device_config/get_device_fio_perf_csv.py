import pathlib, argparse 
from mascots.experiments.DeviceAnalysis import DeviceAnalysis 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance statistics \
        file for a given directory containig device measurements.")
    parser.add_argument("log_dir", type=pathlib.Path, help="Path to store log files.")
    parser.add_argument("perf_csv_path", type=pathlib.Path, help="Path to the performance CSV file.")
    args = parser.parse_args()

    d_analysis = DeviceAnalysis()
    d_analysis.generate_device_fio_perf_csv(args.log_dir, args.perf_csv_path)