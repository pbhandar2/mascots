import argparse, json 
import pathlib  
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib


class ExhaustiveCostExperiment:
    def __init__(self, 
        workload_name, 
        mt_config_file_path,
        experiment_config="../experiment_config.json"):

        self.workload_name = workload_name 
        self.mt_config_file_path = pathlib.Path(mt_config_file_path)
        self.mt_label = self.mt_config_file_path.stem

        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f)

        self.rd_hist_dir = pathlib.Path(self.experiment_config["rd_hist_4k_dir"])
        self.rd_hist_path = self.rd_hist_dir.joinpath("{}.csv".format(workload_name))

        self.cache_unit_size = 256
        self.output_dir = pathlib.Path("/research2/mtc/cp_traces/opt_exhaustive_cost_analysis")


    def opt_entry_to_csv_row(self, opt_entry):
        """ Convery a JSON returned from cost analysis to a CSV entry 
        """

        row = []
        for key in opt_entry:
            if key == "mt_p_size_array" or key == "mt_np_size_array":
                mt_tier_string = "-".join([str(s) for s in opt_entry[key]])
                row.append(mt_tier_string)
            else:
                value = opt_entry[key]
                if type(value) == float:
                    row.append(format(value, '.4f'))
                else:
                    row.append(str(value))
        return ",".join(row)


    def write_opt_entry_to_file(self, opt_entry, file_path):
        """ Write the OPT entry into a CSV file for analysis 
        """

        opt_row_string = self.opt_entry_to_csv_row(opt_entry)
        with file_path.open("a+") as f:
            f.write("{}\n".format(opt_row_string))


    def run(self):
        """ Run the list of mt configurations for user specified workload
            for the given RD histogram and MT cache. 
        """

        profiler = RDHistProfiler(self.rd_hist_path)
        for wb_opt_entry, wt_opt_entry in profiler.exhaustive_cost_analysis([self.mt_config_file_path], 
            self.cache_unit_size):
            """ The main directory containing all results is "/research2/mtc/cp_traces/exhaustive_cost_analysis"
                - one subdirectory each for an MT config (e.g. D1_S1_H1)
                    - one subdirectory each for write-back and write-through 
                        - output file stored as *workload_name*.csv 
            """

            device_data_dir = pathlib.Path(self.output_dir.joinpath(wb_opt_entry["mt_label"]))
            device_data_dir.mkdir(exist_ok=True)

            wb_dir = device_data_dir.joinpath("wb")
            wt_dir = device_data_dir.joinpath("wt")
 
            wb_dir.mkdir(exist_ok=True)
            wt_dir.mkdir(exist_ok=True)

            wb_output_path = wb_dir.joinpath(self.rd_hist_path.name)
            wt_output_path = wt_dir.joinpath(self.rd_hist_path.name)

            self.write_opt_entry_to_file(wb_opt_entry, wb_output_path)
            self.write_opt_entry_to_file(wt_opt_entry, wt_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform exhaustive cost analysis on a workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("--mt_config_dir", 
        default=pathlib.Path("/home/pranav/mtc/mt_config/2"),
        type=pathlib.Path,
        help="Directory containing MT configuration files")
    parser.add_argument("--mt_label", 
        default="D1_S1_H1",
        type=str,
        help="Label of the MT cache")
    args = parser.parse_args()

    mt_config_file_path = args.mt_config_dir.joinpath("{}.json".format(args.mt_label))
    experiment = ExhaustiveCostExperiment(args.workload_name, mt_config_file_path)
    experiment.run()