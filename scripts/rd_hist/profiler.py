import sys 
import numpy as np 
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler

DEFAULT_OUTPUT_PATH = "rd_hist_stats.csv"
 
if __name__ == "__main__":
    p = RDHistProfiler(sys.argv[1])

    hits = p.get_exclusive_cache_hits(5, 10)

    print(np.sum(hits))

    #devices = np.array([[0.0000627, 0.0000627], [0.158, 0.370], [5.5, 5.5]])
    devices = np.ones((3,2))
    p.get_mt_mean_latency(5, 10, devices)

    # output_file = DEFAULT_OUTPUT_PATH
    # if len(sys.argv) > 2:
    #     output_file = sys.argv[2]
    # p.write_block_stats_to_file(output_file)