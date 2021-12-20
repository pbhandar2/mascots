from abc import ABC, abstractmethod

class Reader(ABC):

    def __init__(self, trace_file_path):
        self.trace_file_handle = open(trace_file_path, "r")
    
    @abstractmethod
    def get_next_block_req(self,):
        pass 

    def __exit__(self, exc_type, exc_value, traceback):
        self.trace_file_handle.close()
