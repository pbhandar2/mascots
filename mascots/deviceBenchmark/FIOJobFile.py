import pathlib 

class FIOJobFile:
    def __init__(self, name, io_dir, io_type, seq_percent, req_size, 
        total_io_size, filesize, direct=1, ioengine="psync"):
        """
        The class represents an FIO job file. It allows for creation 
        and modification of FIO parameters and write it out as a jobfile. 

        ...

        Attributes
        ----------
        name : str
            name of the benchmark 
        io_dir : str
            directory where the IO is targeted 
        io_type : str
            the type of IO (read/write)
        seq_percent : int 
            percentage of IO that are sequential 
        req_size : int 
            size of each IO request 
        total_io_size : int
            the total IO performed in the job 
        filesize : int
            the size of a file in GB 
        direct : int, optional 
            flag to set DIRECT I/O (default is 1)
        ioengine : str, optional 
            ioengine parameters in FIO (default is psync)

        Methods
        -------
        write_to_file(self, output_path)
            Write the FIO parameters as an FIO jobfile. 
        """

        self.name = name
        self.io_dir = pathlib.Path(io_dir)
        if io_type == "r":
            self.rw = "randread"
        elif io_type == "w":
            self.rw = "randwrite"
        else:
            pass 

        self.percentage_random = 100 - seq_percent
        self.req_size = req_size 
        self.total_io_size = total_io_size
        self.filesize = filesize 
        self.direct = direct
        self.ioengine = ioengine

    def write_to_file(self, output_path):
        """ Write the parameters as an FIO jobfile. 

        Parameters
        ----------
        output_path : str
            Output path of the job file 
        """

        with open(output_path, "w+") as f:
            f.write("[{}]\n".format(self.name))
            f.write("directory={}\n".format(self.io_dir.resolve()))
            f.write("rw={}\n".format(self.rw))
            f.write("bs={}KB\n".format(self.req_size))
            f.write("ioengine={}\n".format(self.ioengine))
            f.write("io_size={}MB\n".format(self.total_io_size))
            f.write("filesize={}GB\n".format(self.filesize))
            f.write("percentage_random={}\n".format(self.percentage_random))
            f.write("direct={}\n".format(self.direct))