import sys 
import pathlib 
from mascots.deviceBenchmark.FIOJob import FIOJob
 
if __name__ == "__main__":
    # Run the user specified job file 
    benchmark = FIOJob(sys.argv[1])
    benchmark.run()