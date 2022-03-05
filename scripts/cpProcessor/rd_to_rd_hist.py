import sys 
import pathlib 
from mascots.blockReader.RDTraceReader import RDTraceReader

if __name__ == "__main__":
    reader = RDTraceReader(sys.argv[1])
    reader.generate_rd_hist_file(sys.argv[2])