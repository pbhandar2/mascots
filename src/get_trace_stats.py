import argparse 
import pathlib 
from mascots.traceAnalysis.traceAnalysis import TraceAnalysis

# get the block trace stats for each block trace 
block_trace_dir = pathlib.Path("/research/file_system_traces/cp_traces/FAST21/block_4k/")
output_file = pathlib.Path("block_read_write_stats.csv")
headers = ["workload", "read_count", "write_count", "total_count", "read_ws", "write_ws", "total_ws", "time"]


def get_block_trace_read_write_stats(block_trace_path):
    t = TraceAnalysis(block_trace_path)
    return t.get_read_write_stats()


def main():
    with output_file.open("w+") as f:
        f.write("{}\n".format(",".join(headers)))
        for block_trace_path in block_trace_dir.iterdir(): 
            print("Processing {}".format(block_trace_path))
            stats = get_block_trace_read_write_stats(block_trace_path)
            f.write("{}\n".format(",".join([str(stats[h]) for h in headers])))


if __name__ == "__main__":
    main()