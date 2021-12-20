from mascots.blockReader.Reader import Reader

class CPReader(Reader): 

    def __init__(self, block_trace_path):
        self.block_trace_path = block_trace_path 
        self.block_trace_handle = open(block_trace_path, "r")
        self.line = None # stores the previous line 


    def get_next_block_req(self):
        """ It returns a dict of the attributes of the next block request. 
        """
        self.line = self.block_trace_handle.readline().rstrip()
        block_req = {}
        if self.line:
            block_req["ts"], block_req["lba"], block_req["op"], block_req["size"] = self.line.split(",")
            return block_req
        return block_req