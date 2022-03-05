from mascots.blockReader.Reader import Reader

class CPReader(Reader): 
    """
    The class to read the CSV version of CloudPhysics traces 

    ...

    Attributes
    ----------
    trace_path : Path
        Path object to the path of the CSV CloudPhysics trace 
    
    Methods
    -------
    get_next_block_req(self)
        Get JSON object comprising of attributes and values of the next block request 
    generate_page_trace(self, page_trace_path, page_size, block_size)
        Generate a page trace from the block file for a specified block and page size 
    """

    def __init__(self, trace_path):
        """
        Parameters
        ----------
        trace_path : str 
            path of the page trace 
        """

        super().__init__(trace_path)
        self.line = None # stores the previous line 


    def get_next_block_req(self):
        """ Return a dict of block request attributes and values 

        Return 
        ------
        block_req : dict 
            dict with block request attributes and values 
        """
        
        self.line = self.trace_file_handle.readline().rstrip()
        block_req = {}
        if self.line:
            split_line = self.line.split(",")
            block_req["ts"] = int(split_line[0])
            block_req["lba"] = int(split_line[1])
            block_req["op"] = split_line[2]
            block_req["size"] = int(split_line[3])
            return block_req
        return block_req

    
    def reset(self):
        self.trace_file_handle.seek(0)


    def __add__(self, reader2, output_path):
        self.reset()
        reader2.reset()
        out_handle = open(output_path, "w+")

        reader1_req = self.get_next_block_req()
        reader1_start_time = reader1_req["ts"]

        reader2_req = reader2.get_next_block_req()
        reader2_start_time = reader2_req["ts"]

        if reader1_start_time < reader2_start_time:
            out_handle.write(",".join())






        