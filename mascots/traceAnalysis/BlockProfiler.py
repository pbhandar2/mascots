import pathlib, math 

from mascots.blockReader.CPReader import CPReader

# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass

class NoDataFromReader(Error):
    """Raised when the first line from reader is empty"""
    pass


class BlockProfiler:

    def __init__(self, block_trace_path, block_size=512, ts_second_scaler=1000000):
        self.path = pathlib.Path(block_trace_path)
        self.reader = CPReader(block_trace_path)
        self.block_size = block_size
        self.ts_second_scaler = ts_second_scaler

        # workload attributes
        self.start_time = None
        self.io_count = 0
        self.read_count = 0 
        self.write_count = 0 
        self.total_read_size = 0 
        self.total_write_size = 0 

        self.seq_run = 0 
        self.seq_run_count = 0 
        self.random_access_count = 0 
        self.total_jump_distance = 0 
        self.max_seq_run = 0 

        # current request attribute 
        self.cur_time = None
        self.lba = None
        self.op = 0
        self.size = 0 

        # previous request attribute
        self.prev_time = None 
        self.prev_offset_end = -1 


    def generate_report(self, report_dir):
        """ Generates a report of the block trace and stores 
            it in a directory. 
        """
        pass


    def random_access(self):
        if self.seq_run > 1:
            self.seq_run_count += self.seq_run
            if self.max_seq_run < self.seq_run_count:
                self.max_seq_run = self.seq_run_count

        self.random_access_count += 1
        self.seq_run = 1 

    
    def seq_access(self):
        # sequetial 
        self.seq_run += 1


    def record_io(self, block_req):
        self.io_count += 1
        self.cur_time = int(block_req['ts'])
        self.lba = int(block_req['lba'])
        self.op = block_req['op']
        self.size = int(block_req['size'])

        if self.op == "r":
            self.read_count += 1
            self.total_read_size += self.size
        elif self.op == "w":
            self.write_count += 1
            self.total_write_size += self.size 

        jump_distance = abs(self.lba * self.block_size - self.prev_offset_end)
        self.total_jump_distance += jump_distance
        if (jump_distance == 0):
            self.seq_access()
        else:
            self.random_access()

        self.prev_time = self.cur_time
        self.prev_offset_end = self.lba * self.block_size + self.size


    def io_done(self):
        self.seq_access() # check if we left any sequetial runs uncounted when IO was done 


    def run(self):
        block_req = self.reader.get_next_block_req()
        try:
            self.start_time = int(block_req['ts'])
        except NoDataFromReader:
            print("No data received from Reader. The file could be empty or the Reader needs to be reset to start of file.")

        while block_req:
            self.record_io(block_req)
            block_req = self.reader.get_next_block_req()
        self.io_done()


    def write_block_stats_to_file(self, output_file):
        total_io = self.total_write_size+self.total_read_size
        with open(output_file, "a+") as o:
            o.write("{},{:.2f},{:.2f},{:.3f},{:.3f},{},{},{:.3f},{},{:.2f}\n".format(
                self.path.stem, # workload tag 
                self.io_count/1000000, # number of requests (Millions)
                total_io/(1024*1024*1024), # total IO size (GB)
                self.write_count/self.io_count, # write request ratio
                self.total_write_size/total_io, # write IO ratio
                math.ceil(self.total_read_size/self.read_count), # mean read request size 
                math.ceil(self.total_write_size/self.write_count), # mean write request size,
                self.seq_run_count/self.io_count, # sequential access ratio 
                self.max_seq_run,
                (self.cur_time - self.start_time)/(self.ts_second_scaler*60*60) # hours 
            ))


    def get_data_headers(self):
        return [
            "workload",
            "io_count",
            "total_io_size",
            "write_count_ratio",
            "write_io_size_ratio",
            "mean_read_size",
            "mean_write_size",
            "sequential_ratio",
            "max_sequential_run",
            "len"
        ]