import sys 

from mascots.blockReader.Merger import Merger

if __name__ == "__main__":
    merger_json_file = "sample-merger.json"
    merger = Merger(sys.argv[1])
    merger.merge()