import math, pathlib, json 
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler

class CostAnalysis:
    """ This class allows users to run cost analysis on a
        reuse distance histogram. 

    Attributes
    ----------
    rd_hist_path : str
        the path to the reuse distance historgram/workload being used 
    output_path : str
        path to store the analysis results 
    page_size : int (optional)
        the size of page (defaults to 4096)
    write_policy : str (optional)
        write policy (defaults to "wt")
    workload_name : str 
        the name of the workload 
    profiler : RDHistProfiler   
        the profiler to evaluate MT cache configuration on the workload 
    key_list : list 
        list of statistics that the analysis outputs to a file 

    Methods
    -------
    get_mt_name
        get the name of the MT cache from the labels of each cache tier  
    setup_mt_config_output_dir
        check and create directory to output data for a MT cache configuration 
    eval_mt
        evaluate an MT cache for this workload at varying cost values 
    """

    def __init__(self, rd_hist_path, output_path, page_size=4096, write_policy="wt"):
        self.page_size = page_size
        self.write_policy = write_policy
        self.rd_hist_path = pathlib.Path(rd_hist_path)
        self.workload_name = self.rd_hist_path.stem 
        self.output_path = pathlib.Path(output_path)
        assert self.output_dir.is_dir()
        self.profiler = RDHistProfiler(rd_hist_path)
        self.key_list = [
            "cost",
            "st_t1",
            "st_lat",
            "mt_p_t1",
            "mt_p_t2",
            "mt_p_lat",
            "mt_np_t1",
            "mt_np_t2",
            "mt_np_lat"]

    
    def get_mt_name(self, mt_config):
        """ Get the name of the MT cache which is derived from the label of 
            the cache device at each cache tier. 

        Parameters
        ----------
        mt_config : JSON
            a JSON MT configuration 
        Return
        ------
        mt_name : str
            the name of the MT cache of the given configuration 
        """

        return "{}_{}_{}".format(
            mt_config[0]["label"],
            mt_config[1]["label"],
            mt_config[2]["label"]
        )


    def setup_mt_config_output_dir(self, mt_config):
        """ Setup output directory for a MT cache configuration. 

        Parameters
        ----------
        mt_config : JSON
            a JSON MT configuration 
        Return
        ------
        output_dir : str
            the path of the directory to output the cost analysis results 
        """

        mt_name = self.get_mt_name(mt_config)
        mt_output_dir = self.output_path.joinpath(mt_name)
        if not mt_output_dir.exists():
            mt_output_dir.mkdir()
        return mt_output_dir


    def eval_mt(self, mt_config_path, min_cost=1, cost_step=1, max_cost=-1, cache_unit_size=256):
        """ Evaluate an MT configurations at varying cost. 

        Parameters
        ----------
        mt_config_path : str
            path to the MT configuration file 
        min_cost : int (optional)
            min dollar cost to evaluate (Defaults to 1)
        cost_step : int (optional)
            the step size of increasing dollar cost (Defaults to 1)
        max_cost : int (optional)
            the maximum cost to evalaute (Defaults to -1)
        cache_unit_size : int (optional)
            the granularity of cache evaluate (Defaults to 256) 
            for a RD histogram representing 4KB objects, a cache_unit_size 
            of 256 would set the granularity of cache analysis to 4KB *256=1MB. 
            This means that cache can only by whole numbers of MB unit. 5MB exists
            but 5.4MB does not. 
        Return
        ------
        output_dir : str
            the path of the directory to output the cost analysis results 
        """

        mt_config_path = pathlib.Path(mt_config_path)
        with mt_config_path.open("r") as f:
            mt_config = json.load(f)

        mt_output_dir = self.setup_mt_config_output_dir(mt_config)
        csv_output_path = mt_output_dir.joinpath("{}.csv".format(self.workload_name))
        if max_cost == -1:
            max_cost = math.ceil(len(self.profiler.data)*self.profiler.get_page_cost(mt_config, 0))

        with csv_output_path.open("w+") as csv_handle:
            for cost in range(min_cost, max_cost, cost_step):
                mt_eval_json = self.profiler.cost_eval_exclusive(mt_config, self.write_policy, cost, cache_unit_size)   
                out_string = ",".join([str(mt_eval_json[key] for key in self.key_list)])
                csv_handle.write("{}\n".format(out_string))
