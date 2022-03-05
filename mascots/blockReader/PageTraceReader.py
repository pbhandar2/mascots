from PyMimircache import Cachecow

class PageTraceReader:
    def __init__(self, page_trace_path, lba_index=0, op_index=1, ts_index=2, delimiter=","):
        self.page_trace_path = page_trace_path 
        self.page_trace_handle = open(page_trace_path, "r")

        self.lba_index = lba_index 
        self.op_index = op_index
        self.ts_index = ts_index
        self.delimiter = delimiter 

        self.reader_params = {
            "init_params": {
                    "label": lba_index+1, # the indexing start from 1 
                    "real_time": ts_index+1, 
                    "op": op_index+1, 
                    "delimiter": self.delimiter 
            },
            "label": lba_index+1,
            "delimiter": self.delimiter 
        }

    def generate_rd_trace(self, rd_trace_path):
        mimircache = Cachecow()
        mimircache.csv(self.page_trace_path, self.reader_params)
        profiler = mimircache.profiler(algorithm="LRU")
        rd_list = profiler.get_reuse_distance() 

        with open(self.page_trace_path) as page_handle:
            with open(rd_trace_path, "w+") as rd_handle:
                for rd in rd_list:
                    page_line = page_handle.readline().rstrip()
                    line_split = page_line.split(self.delimiter)
                    op = line_split[self.op_index]
                    rd_handle.write("{},{}\n".format(rd,op))
