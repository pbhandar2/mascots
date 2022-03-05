import argparse, pathlib, json
from asyncore import write
from genericpath import exists 
from mascots.traceAnalysis.MHMRCProfiler import MHMRCProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib


class MaxHMRCCostExperiment:
    """ This class runs OPT cache analysis for a given workload, 
        MT cache and cost. 
    """
    def __init__(self, 
        workload_name, # workload name to identify relevant files 
        mt_config_file_path, # file with device specifications of each tier 
        write_policy, # wb or wt
        alg, # algorithm label
        step_size, # step size used in the algorithm
        allocation_unit=256, # number of pages in a unit cache allocation 
        experiment_config="../experiment_config.json"):

        self.workload_name = workload_name 
        self.mt_config_file_path = pathlib.Path(mt_config_file_path)
        self.mt_label = self.mt_config_file_path.stem
        self.algorithm = alg
        self.step_size = step_size
        self.write_policy = write_policy 

        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f)
        
        self.rd_hist_dir = pathlib.Path(self.experiment_config["rd_hist_4k_dir"])
        self.rd_hist_path = self.rd_hist_dir.joinpath("{}.csv".format(workload_name))

        self.allocation_unit = allocation_unit
        self.setup_output_dir()

    
    def setup_output_dir(self):
        """ Setup the output directory 

        Return 
        ------
        output_dir : pathlib.Path
            the directory to output experiment data 
        """

        main_output_dir = pathlib.Path(self.experiment_config["mhmrc_cost_analysis_dir"])
        """ Main output dir is user specified let say X for now 
            - subdir for each device config (e.g. D1_S1_H1)
                - subdir for write policy (e.g. wb)
                    - subdir for each workload (e.g. w01)
                        - subdir for algo type (e.g. v1)
                            -subdir for each param which here is stepsize 
                                - output file with format *eval count*.csv (e.g. 10.csv)

            example output path: X/D1_S1_H1/wb/w01/v1/10_2.csv 
        """
        out_dir = main_output_dir.joinpath(
            self.mt_label,
            self.write_policy,
            self.workload_name,
            self.algorithm,
            str(self.step_size)
        )
        out_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = out_dir


    def json_out_to_str(self, out):
        """ Convert a JSON returned from cost analysis to a CSV entry 
        """

        row = [str(out[key]) for key in out]
        return ",".join(row)


    def write_json_out_to_file(self, out, file_path):
        """ Write the output into a CSV file for analysis 
        """

        out_string = self.json_out_to_str(out)
        with file_path.open("a+") as f:
            f.write("{}\n".format(out_string))


    def run(self):
        """ Run the cost analysis experiment using Max Hit-Miss Ratio Curve 

        Parameters
        ----------
        step_size : int 
            the step size to use in the algorithm
        """

        max_hmrc_profiler = MHMRCProfiler(self.rd_hist_path)
        if self.algorithm == "v1": 
            for wb_out, wt_out in max_hmrc_profiler.get_opt_form_hmrc(self.mt_config_file_path,
                                    step_size=self.step_size):
                for key in wb_out:
                    output_file_name = "{}_{}.csv".format(self.step_size, key)
                    wb_output_path = self.wb_output_dir.joinpath(output_file_name)
                    wt_output_path = self.wt_output_dir.joinpath(output_file_name)
                    self.write_json_out_to_file(wb_out[key], wb_output_path)
                    self.write_json_out_to_file(wt_out[key], wt_output_path)
        elif self.algorithm == "v2":
            for out_json in max_hmrc_profiler.get_opt_form_hmrc_v2(self.mt_config_file_path,
                                step_size=self.step_size, write_policy=self.write_policy):
                for key in out_json:
                    output_file_name = "{}.csv".format(key)
                    output_path = self.out_dir.joinpath(output_file_name)
                    self.write_json_out_to_file(out_json[key], output_path)


    def run_single(self, mt_config, cost):
        """ Run the cost analysis experiment using Max Hit-Miss Ratio Curve 

        Parameters
        ----------
        step_size : int 
            the step size to use in the algorithm
        """


        max_hmrc_profiler = MHMRCProfiler(self.rd_hist_path)
        max_t1_size = max_hmrc_profiler.mt_config_lib.get_max_tier_size(mt_config, 0, 
                max_hmrc_profiler.rd_hist_profiler.allocation_unit, cost)
        cur_mhmrc = max_hmrc_profiler.get_scaled_mhmrc(max_hmrc_profiler.mhmrc, mt_config, cost)
        cur_mhmrc = cur_mhmrc[:max_t1_size]

        out_dict = max_hmrc_profiler.eval_mhmrc_v2(cur_mhmrc, mt_config, cost, 
                        write_policy=self.write_policy,
                        step_size=self.step_size)

        for key in out_dict:
            output_file_name = "{}.csv".format(key)
            output_path = self.out_dir.joinpath(output_file_name)
            self.write_json_out_to_file(out_dict[key], output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Max Hit-Miss Ratio based generation of cost-efficient MT cache")
    parser.add_argument("workload_name", 
        help="The name of the workload to be evaluted")
    parser.add_argument("--mt_config_dir", 
        default=pathlib.Path("/home/pranav/mtc/mt_config/2"),
        type=pathlib.Path,
        help="Directory containing MT configuration files")
    parser.add_argument("--mt_config_label", 
        default="D1_S1_H1",
        type=str,
        help="Label of the MT cache")
    parser.add_argument("--step_size", 
        default=1,
        type=int,
        help="Step size for analysis")
    parser.add_argument("--alg",
        default="v2",
        type=str,
        help="The version of the algorithm to run")
    parser.add_argument("--write_policy",
        default="wb",
        type=str,
        help="The write-policy to use 'wb' or 'wt'")
    args = parser.parse_args()

    experiment = MaxHMRCCostExperiment(args.workload_name,
        args.mt_config_dir.joinpath("{}.json".format(args.mt_config_label)),
        args.write_policy,
        args.alg, 
        args.step_size)
    experiment.run()