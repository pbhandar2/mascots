import sys 
import pathlib 
import pandas as pd 

BLOCK_TRACE_DIR = pathlib.Path("/research2/mtc/cp_traces/cache_trace/block_4k")
PAGE_SIZE = 4096


def main():
    block_trace_list = BLOCK_TRACE_DIR.glob("*.csv")
    for block_trace_path in block_trace_list :
        df = pd.read_csv(block_trace_path, names=["lba", "op", "ts"])
        read_count = len(df[df["op"]=="r"])
        working_set_size_gb = df["lba"].nunique()*4096/(1024*1024*1024)
        lba_stats = df["lba"].describe()
        lba_range_gb = (lba_stats["max"] - lba_stats["min"])*4096/(1024*1024*1024)
        read_working_set_size_gb = df[df["op"]=="r"]["lba"].nunique()*4096/(1024*1024*1024)
        write_working_set_size_gb = df[df["op"]=="w"]["lba"].nunique()*4096/(1024*1024*1024)

        print("{},{},{},{},{},{},{}".format(
            block_trace_path.stem,
            len(df),
            read_count/len(df),
            lba_range_gb,
            working_set_size_gb,
            read_working_set_size_gb,
            write_working_set_size_gb
        ))


if __name__ == "__main__":
    main()





