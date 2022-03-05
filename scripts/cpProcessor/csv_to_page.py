import sys 
import pathlib 
from mascots.blockReader.CPReader import CPReader

if __name__ == "__main__":
    reader = CPReader(sys.argv[1])
    reader.generate_page_trace(sys.argv[2], 4096, 512)