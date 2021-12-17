import pandas as pd 

def load_block_trace(block_trace_path):
    return pd.read_csv(block_trace_path, names=["lba", "op", "ts"])

