import argparse, json, time, logging, sys 
import pathlib  
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib

logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

class CostAnalysis:
    def __init__(self, 
            workload_name, 
            cost,
            num_tiers = 2,
            allocation_unit = 256,
            experiment_config="../experiment_config.json"):

        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f)

        self.workload_name = workload_name 
        self.cost = cost 
        self.device_dir = pathlib.Path(self.experiment_config["device_config_dir"]).joinpath(str(num_tiers))
        self.mt_list = self.experiment_config["priority_2_tier_mt_label_list"]

        self.rd_hist_dir = pathlib.Path(self.experiment_config["rd_hist_4k_dir"])
        self.rd_hist_path = self.rd_hist_dir.joinpath("{}.csv".format(workload_name))
        self.profiler = RDHistProfiler(self.rd_hist_path)

        self.allocation_unit = allocation_unit
        self.main_output_dir = pathlib.Path(self.experiment_config["ex_cost_analysis_dir"])


    def get_output_path(self, mt_label, write_policy):
        wb_out_dir = self.main_output_dir.joinpath(mt_label, write_policy)
        wb_out_dir.mkdir(exist_ok=True, parents=True)
        return wb_out_dir.joinpath("{}.csv".format(self.workload_name))


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

        for mt_label in self.mt_list:
            logging.info("Computing workload {}, cost {}, MT {}".format(self.workload_name,
                                                                        self.cost, 
                                                                        mt_label))
            start = time.perf_counter()
            mt_config_file_path = self.device_dir.joinpath("{}.json".format(mt_label))
            with open(mt_config_file_path) as f:
                mt_config = json.load(f)
            
            wb_out_dict, wt_out_dict = self.profiler.two_tier_optimized_cost_analysis(mt_config, self.cost)
            wb_out_path = self.get_output_path(mt_label, "wb")
            wt_out_path = self.get_output_path(mt_label, "wt")

            self.write_opt_entry_to_file(wb_out_dict, wb_out_path)
            self.write_opt_entry_to_file(wt_out_dict, wt_out_path)

            end = time.perf_counter()
            logging.info("workload: {} cost: {}, MT: {} Done!, Time Taken: {}".format(
                self.workload_name,
                self.cost, 
                mt_label,
                end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Perform cost analysis given a workload and cost")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("cost", type=int, help="The cost limit of the cache ")
    args = parser.parse_args()

    experiment = CostAnalysis(args.workload_name, args.cost)
    experiment.run()