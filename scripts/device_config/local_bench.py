import sys 
import pathlib
from typing import IO 
from mascots.deviceBenchmark.LocalBench import LocalBench

JOB_FILE_DIR = pathlib.Path("/home/pranav/tmp_jobs")
IO_DIR = pathlib.Path("/home/pranav/tmp_io")
LOG_DIR = pathlib.Path("/home/pranav/tmp_log")

if not JOB_FILE_DIR.is_dir():
    pathlib.Path.mkdir(JOB_FILE_DIR)
if not IO_DIR.is_dir():
    pathlib.Path.mkdir(IO_DIR)
if not LOG_DIR.is_dir():
    pathlib.Path.mkdir(LOG_DIR)
 
if __name__ == "__main__":
    benchmark = LocalBench("test", JOB_FILE_DIR, IO_DIR, LOG_DIR)
    benchmark.run()