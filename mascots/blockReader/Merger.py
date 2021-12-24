import pathlib, json, math 
import numpy as np 
from mascots.blockReader.CPReader import CPReader

class Merger:

    def __init__(self, config_file):
        """
        This class merges multiple block traces.

        ...

        Attributes
        ----------
        config : obj
            JSON configuration of the merger 
        num_files : int
            the number of files being merged 
        path_list : list 
            list of block traces paths being merged 
        reader_list : list
            list of Reader object for each block trace 
        req_list : list 
            list of current request for each block trace 
        start_time_list : list 
            list of start time of each block trace 
        cur_time_list : list 
            list of time of the next request of each block trace 
        offset_list : list 
            list of global starting offset of each block trace 
        range_gb : int 
            range of IO in GB 
        range_list : list
            list of ranges of each block trace 
        min_lba_list : list 
            list of minimum LBA accessed in each block trace 

        Methods
        -------
        sanity_check(self)
            Check the paths and load the initial data. 
        """

        self.config = json.load(open(config_file))
        self.num_files = len(self.config["file_list"])
        self.path_list = [None] * self.num_files
        self.reader_list = [None] * self.num_files
        self.req_list = [None] * (self.num_files)
        self.start_time_list = [math.inf] * self.num_files
        self.cur_time_list = [math.inf] * self.num_files
        self.offset_list = [0] * self.num_files

        self.range_gb = 0 
        self.range_list = [0] * self.num_files
        self.min_lba_list = [-1] * self.num_files
        
        self.sanity_check()


    def sanity_check(self):
        for index, file_info in enumerate(self.config["file_list"]):
            cur_path = pathlib.Path(file_info["path"])
            assert(cur_path.exists())
            self.path_list[index] = cur_path

            self.offset_list[index] = (self.range_gb*1024*1024*1024)/self.config["lba_size"]
            self.range_gb += file_info["range_gb"]
            self.range_list[index] = int(file_info["range_gb"])
            self.min_lba_list[index] = int(file_info["min_lba"])
            
            reader = CPReader(file_info["path"])
            cur_req = reader.get_next_block_req()
            self.start_time_list[index] = int(cur_req["ts"]) if cur_req else math.inf
            self.req_list[index] = cur_req
            self.cur_time_list[index] = 0
            self.reader_list[index] = reader
            

    def update_reader_data(self, index):
        cur_req = self.reader_list[index].get_next_block_req()
        if not cur_req:
            self.req_list[index] = {}
            self.cur_time_list[index] = math.inf 
        else:
            self.req_list[index] = cur_req
            self.cur_time_list[index] = int(cur_req["ts"]) - self.start_time_list[index]

            
    def merge(self):
        with open(self.config["output_path"], "w+") as f:
            min_time = min(self.cur_time_list)
            min_time_index = self.cur_time_list.index(min_time)
            while min_time != math.inf:
                cur_req = self.req_list[min_time_index]
                adjusted_ts = self.cur_time_list[min_time_index]
                adjusted_lba = int(self.offset_list[min_time_index] + int(cur_req["lba"]) - self.min_lba_list[min_time_index])
                line = "{},{},{},{}\n".format(adjusted_ts, adjusted_lba, cur_req["op"], cur_req["size"])
                f.write(line)
                self.update_reader_data(min_time_index)
                min_time = min(self.cur_time_list)
                min_time_index = self.cur_time_list.index(min_time)




