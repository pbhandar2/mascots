from abc import ABC, abstractmethod

class Reader(ABC):

    def __init__(self, trace_file_path):
        """
        The abstract Reader class. 

        ...

        Attributes
        ----------
        trace_file_handle : obj
            handle of the open block trace file 

        Methods
        -------
        get_next_block_req(self)
            Get the next block request. 
        """

        self.trace_file_handle = open(trace_file_path, "r")
    
    @abstractmethod
    def get_next_block_req(self):
        pass 

    def __exit__(self, exc_type, exc_value, traceback):
        self.trace_file_handle.close()
