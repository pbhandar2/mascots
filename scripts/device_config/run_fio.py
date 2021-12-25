import sys 
import pathlib 
from mascots.deviceBenchmark.FIOJob import FIOJob
 
if __name__ == "__main__":
    # run the specified FIO job file 
    benchmark = FIOJob(sys.argv[1])
    benchmark.run()