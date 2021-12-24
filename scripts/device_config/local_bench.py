import sys 
import pathlib, argparse
from mascots.deviceBenchmark.LocalBench import LocalBench

JOB_DIR = pathlib.Path("/home/pranav/tmp_jobs")
IO_DIR = pathlib.Path("/home/pranav/tmp_io")
LOG_DIR = pathlib.Path("/home/pranav/tmp_log")

def create_output_dirs(job_dir, log_dir, io_dir):
    if not job_dir.is_dir():
        pathlib.Path.mkdir(job_dir)
    if not io_dir.is_dir():
        pathlib.Path.mkdir(io_dir)
    if not log_dir.is_dir():
        pathlib.Path.mkdir(log_dir)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a set of FIO jobs to measure \
        single threaded DIRECT IO performance under varying IO request size.")
    parser.add_argument("name", help="Name to identify the benchmark run.")
    parser.add_argument("--job_dir", type=pathlib.Path, 
        default=JOB_DIR, help="Path to store the job files.")
    parser.add_argument("--log_dir", type=pathlib.Path, 
        default=LOG_DIR, help="Path to store log files.")
    parser.add_argument("--io_dir", type=pathlib.Path, 
        default=IO_DIR, help="Path to perform IO. The device to be benchmark should \
            be mounted to this path.")
    parser.add_argument("--is_ram_disk", action='store_true', 
        help="Whether the path is a RAM disk.")
    args = parser.parse_args()

    create_output_dirs(args.job_dir, args.log_dir, args.io_dir)
    benchmark = LocalBench(args.name, args.job_dir, args.io_dir, args.log_dir, is_ram_disk=args.is_ram_disk)
    benchmark.run()