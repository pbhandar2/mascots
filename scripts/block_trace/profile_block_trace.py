import sys 
from mascots.traceAnalysis.BlockProfiler import BlockProfiler

DEFAULT_OUTPUT_PATH = "block_stats.csv"
 
if __name__ == "__main__":
    p = BlockProfiler(sys.argv[1])
    p.run()

    output_file = DEFAULT_OUTPUT_PATH
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    p.write_block_stats_to_file(output_file)