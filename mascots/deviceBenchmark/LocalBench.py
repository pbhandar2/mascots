import pathlib 
import logging
logging.basicConfig(level=logging.INFO)

from mascots.deviceBenchmark.FIOJobFile import FIOJobFile
from mascots.deviceBenchmark.FIOJob import FIOJob

class LocalBench:
    def __init__(self, name, job_dir, io_dir, log_dir, is_ram_disk=False):
        """
        The class represents a single benchmark run. The class should run 
        benchmarks based on the input provided by the user, collect output
        logs, analyze and output human readable information and data. 

        ...

        Attributes
        ----------
        name : str
            name of the benchmark 
        job_dir : str
            directory to store job files created for the benchmark 
        io_dir : str
            directory where the IO is targeted 
        log_dir : str
            directory where the output FIO logs are stored 
        is_ram_disk : bool, optional 
            flag to identify RAM disk evaluation (this disables direct I/O)

        Methods
        -------
        generate_jobfile(self, job_path, io_type="r", req_size=4096, seq_percent=50)
            Generates a FIO jobfile. 
        run(self, mode="rwonly")
            Run the local benchmark. 
        """

        self.read_req_size_list = [4, 8, 12, 16, 24, 32, 48, 64, 128]
        self.write_req_size_list = [4, 8, 12, 16, 24, 32, 48, 64, 128]
        self.seq_percent_list = [0, 25, 50, 75, 100]
        self.filesize_gb = 10
        self.total_io_mb = 500
        
        self.name = name
        self.io_dir = pathlib.Path(io_dir)
        self.log_main_dir = pathlib.Path(log_dir)
        self.job_main_dir = pathlib.Path(job_dir)

        # setup subdirectory for output based on the name 
        self.job_dir = self.job_main_dir.joinpath(self.name)
        if not self.job_dir.is_dir():
            pathlib.Path.mkdir(self.job_dir)

        self.log_dir = self.log_main_dir.joinpath(self.name)
        if not self.log_dir.is_dir():
            pathlib.Path.mkdir(self.log_dir)

        self.is_ram_disk = is_ram_disk


    def generate_jobfile(self, job_path, io_type="r", req_size=4096, seq_percent=50):
        """ Generates a FIO jobfile. 

        Parameters
        ----------
        job_path : str
            Output path of the job file 

        io_type : str, optional
            Type of IO to perform (default is "w")

        req_size : int, optional
            Size of each IO request in bytes (default is 4096)

        seq_percent : int, optional
            Percentage of IO that is sequential (default is 50)
        """

        jobfile = FIOJobFile(self.name, self.io_dir, io_type, seq_percent, req_size, self.total_io_mb, self.filesize_gb)
        jobfile.write_to_file(job_path)


    def run(self, mode="rwonly"):
        """ Run the benchmark. 

        Parameters
        ----------
        mode : str, optional
            Mode defines the type of read, write IO performed in a single run of benchmark. Types:
                1. "rwonly": run only read and write during a 
                    single benchmark run. 
        """

        if mode == "rwonly":
            for io_type in ["r", "w"]:
                req_size_list = self.read_req_size_list if io_type == "r" else self.write_req_size_list
                for req_size in req_size_list:
                    for seq_percent in self.seq_percent_list:
                        file_name = "{}_{}_{}.job".format(io_type, req_size, seq_percent)
                        job_path = self.job_dir.joinpath(file_name)
                        self.generate_jobfile(job_path, io_type=io_type, req_size=req_size, seq_percent=seq_percent)      
                        job = FIOJob(job_path)
                        output_path = self.log_dir.joinpath(file_name.replace(".job", ".json"))
                        out = job.run(log_output_file=output_path)

                        if out == 0:
                            logging.info("Done! Evaluating {}".format(job_path))
                        else:
                            logging.warning("Failed! Evaluating {}".format(job_path))


                        
