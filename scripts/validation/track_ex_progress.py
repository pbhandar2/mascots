import argparse, pathlib, json
from tabulate import tabulate
import pandas as pd 
import numpy as np 

from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON

class Validator:
    def __init__(self,
            experiment_config="../experiment_config.json"):

        self.experiment_config_path = experiment_config
        with open(self.experiment_config_path, "r") as f:
            self.experiment_config = json.load(f)

        self.max_cost_df = pd.read_csv("../workload_analysis/max_cost.csv", 
                                        delimiter=" ",
                                        names=["workload", "max_cost"])
        self.ex_out_dir = pathlib.Path(self.experiment_config["ex_cost_analysis_dir"])


    def load_ex_data(self, ex_out_path):
        """ Load the output from exhaustive search 

            Parameters
            ----------
            ex_out_path : pathlib.Path 
                the path to the file with output from exhaustive search 
            
            Return 
            ------
            df : pd.DataFrame 
                the dataframe containing the output 
        """

        return pd.read_csv(ex_out_path, names=OPT_ROW_JSON)

    
    def track_ex_progress_cp(self, mt_label="D1_S1_H1"):
        """ Track the number of points that are left to be 
            processed and identify and remove duplicates. 

            Parameters
            ----------
            mt_label : str 
                label to identify the MT cache 
        """

        progress_detail_array = []
        main_dir = self.ex_out_dir.joinpath(mt_label, "wb")
        for ex_out_file in main_dir.iterdir():
            workload_name = ex_out_file.stem 
            max_cost_row = self.max_cost_df[self.max_cost_df["workload"]==workload_name]

            try:
                assert(len(max_cost_row) > 0)
            except AssertionError:
                print("No data found for workload {}".format(workload_name))

            df = pd.read_csv(ex_out_file, names=OPT_ROW_JSON)
            cost_evals_remaining = max_cost_row.iloc[0]["max_cost"]-len(df)
            if cost_evals_remaining > 0:
                progress_detail_array.append([workload_name, cost_evals_remaining])
            else:
                if df["cost"].duplicated().any():
                    df = df.drop_duplicates()
                    df = df[df["num_eval"]>0].sort_values(by=["cost"])
                    df.to_csv(ex_out_file, header=False, index=False)

            max_cost_entry = df[df["cost"]==df["max_cost"]]
            if len(max_cost_entry) == 1:
                assert(max_cost_entry.iloc[0]["mt_opt_flag"]==0)
        
        print(tabulate(sorted(progress_detail_array, key=lambda k: k[1])))
        print("Remaining: {}".format(np.array([_[1] for _ in progress_detail_array]).sum()))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track progress of exhaustive search algorithms")
    args = parser.parse_args()
    validator = Validator()
    validator.track_ex_progress_cp()