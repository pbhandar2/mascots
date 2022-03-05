import sys 
import pathlib 
from mascots.blockReader.PageTraceReader import PageTraceReader

if __name__ == "__main__":
    reader = PageTraceReader(sys.argv[1])
    reader.generate_rd_trace(sys.argv[2])