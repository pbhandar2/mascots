import pathlib 
import subprocess 

class FIOJob:
    def __init__(self, fio_job_file):
        """
        The class runs a given FIO job and returns the output. 

        ...

        Attributes
        ----------
        fio_job_file : str
            Path of the FIO job file to be run 

        Methods
        -------
        run(self, log_output_file="")
            Run the FIO job and write the output to the provided file. 
        """

        self.job_file = pathlib.Path(fio_job_file)

    def run(self, log_output_file=""):
        fio_cmd = ["fio", self.job_file.resolve()]
        if log_output_file != "":
            fio_cmd.append("--output-format=json")
            fio_cmd.append("--output={}".format(log_output_file))
        return subprocess.call(fio_cmd, stdin=None, stdout=None, stderr=None, shell=False)
        

