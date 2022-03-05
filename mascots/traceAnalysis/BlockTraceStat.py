import math 

class BlockTraceStat:
    def __init__(self, block_size=512, ts_scaler=1e6):
        self.block_size = block_size 
        self.ts_scaler = ts_scaler

        self.io_count = 0 
        self.read_count = 0 
        self.write_count = 0 

        self.io_size = 0 
        self.read_io_size = 0
        self.write_io_size = 0

        self.min_lba = -1 
        self.max_lba = -1 

        self.read_lba_set = set()
        self.write_lba_set = set() 

        self.prev_offset_end = -1
        self.seq_count = 0 
        self.total_jump_distance = 0 

        self.start_time = -1 
        self.cur_time = -1 


    def get_range(self):
        return self.block_size * (self.max_lba - self.min_lba)

    
    def get_write_request_ratio(self):
        if self.write_count == 0:
            return 0.0
        return self.write_count/self.io_count 


    def get_write_io_size_ratio(self):
        if self.write_count == 0:
            return 0.0 
        return self.write_io_size/self.io_size

    def get_seq_ratio(self):
        if self.io_count == 0:
            return 0.0
        return self.seq_count/self.io_count 


    def get_read_working_set_size(self):
        return len(self.read_lba_set)


    def get_write_working_set_size(self):
        return len(self.write_lba_set)


    def get_read_write_working_set_overlap(self):
        return len(self.read_lba_set.intersection(self.write_lba_set))


    def get_mean_io_req_size(self):
        if self.io_count == 0:
            return 0.0 
        return self.io_size/self.io_count 


    def get_mean_read_req_size(self):
        if self.read_count == 0:
            return 0.0 
        return self.read_io_size/self.read_count 


    def get_mean_write_req_size(self):
        if self.write_count == 0:
            return 0.0 
        return self.write_io_size/self.write_count 


    def record_block_req(self, op, size, lba, ts):
        self.io_count += 1 
        self.io_size += size 
        if op == 0:
            self.read_count += 1 
            self.read_io_size += size 
        else:
            self.write_count += 1 
            self.write_io_size += size 
        
        if self.min_lba == -1 or self.min_lba > lba:
            self.min_lba = lba 

        if self.max_lba == -1 or self.max_lba < lba:
            self.max_lba = lba 

        if self.start_time == -1:
            self.start_time = ts
        self.cur_time = ts 

        assert size % self.block_size == 0 

        num_lba_accessed = int(size/self.block_size)
        for i in range(num_lba_accessed):
            if op == 0:
                self.read_lba_set.add(lba+i)
            else:
                self.write_lba_set.add(lba+i)

        start_offset = lba*self.block_size
        if self.prev_offset_end == start_offset:
            self.seq_count += 1 

        self.total_jump_distance += abs(start_offset - self.prev_offset_end)
        self.prev_offset_end = start_offset + size 

    
    def write_to_file(self, output_file, workload_name="NA"):
        with open(output_file, "a+") as o:
            o.write(self.get_render_string(workload_name=workload_name))
    

    def render(self, workload_name="NA"):
        print(self.get_render_string(workload_name=workload_name))


    def get_render_string(self, workload_name="NA"):
        return "{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.3f},{:.3f},{:.3f}\n".format(
            workload_name, # workload name
            self.io_count/1e6, # number of requests (Millions)
            self.get_range()/1e9, # diff between max and min offset accessed (GB)
            (self.get_read_working_set_size()*512)/1e9, # read working set size (GB)
            (self.get_write_working_set_size()*512)/1e9, # write working set size (GB)
            (self.get_read_write_working_set_overlap()*512)/1e9, # read/write workign set size overlap (GB)
            self.io_size/1e9, # total IO size (GB)
            self.get_write_request_ratio(), # write request ratio
            self.get_write_io_size_ratio(), # write IO ratio
            math.ceil(self.get_mean_read_req_size()), # mean read request size (bytes)
            math.ceil(self.get_mean_write_req_size()), # mean write request size (bytes)
            self.get_seq_ratio(), # sequential access ratio 
            self.total_jump_distance/(self.io_count*1e6), # average jump distance (MB)
            (self.cur_time - self.start_time)/(self.ts_scaler*60*60) # hours 
        )


    def get_data_headers(self):
        return [
            "workload",
            "io_count",
            "range",
            "read_ws",
            "write_ws",
            "overlap_ws",
            "total_io",
            "write_ratio",
            "write_io_ratio",
            "mean_read_size",
            "mean_write_size",
            "seq_ratio",
            "mean_jump_distance",
            "trace_len"
        ]
        
        
            