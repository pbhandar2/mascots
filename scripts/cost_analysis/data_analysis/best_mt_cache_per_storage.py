import argparse, pathlib, json
from statistics import mean 
import pandas as pd  
import numpy as np 

from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

class PerMTCache:
    def __init__(self,
        experiment_config="../../experiment_config.json"):
        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f)
        self.ex_dir = pathlib.Path(self.experiment_config["ex_cost_analysis_dir"])

    def run(self):
        for mt_label in self.experiment_config["priority_2_tier_mt_label_list"]:
            for write_policy in ["wb", "wt"]:
                
                mt_p_percent_reduction_array = []
                mt_np_percent_reduction_array = []
                p_vs_np_percent_reduction_array = []
                max_p_vs_np_entry = None 
                count_dict = {
                    "st": 0,
                    "mt_p": 0,
                    "mt_np": 0 
                }
                total_lat = 0.0
                total_st_lat = 0.0 
                total_req = 0 
                workload_dir = self.ex_dir.joinpath(mt_label, write_policy)
                num_workloads = len(list(workload_dir.iterdir()))
                workload_index = 0 
                for workload_path in workload_dir.iterdir():
                    workload_name = workload_path.stem 
                    workload_index += 1 
                    #print("loading {}, {}/{}".format(workload_name, workload_index, num_workloads))
                    df = pd.read_csv(workload_path, names=OPT_ROW_JSON)
                    df = df.drop_duplicates()
                    for _, row in df.iterrows():
                        st_latency = row["st_latency"]
                        mt_p_latency = row["mt_p_latency"]
                        mt_np_latency = row["mt_np_latency"]
                        min_mt_latency = min(mt_p_latency, mt_np_latency)
                        total_lat += min(st_latency, mt_p_latency, mt_np_latency)
                        total_st_lat += st_latency
                        total_req += 1

                        if min_mt_latency < st_latency:
                            if mt_p_latency < mt_np_latency:
                                percent_latency_reduction = 100*(st_latency - mt_p_latency)/st_latency
                                mt_p_percent_reduction_array.append(percent_latency_reduction)
                                count_dict["mt_p"] += 1
                            else:
                                percent_latency_reduction = 100*(st_latency - mt_np_latency)/st_latency
                                p_vs_np_percent_latency_reduction = 100*(mt_p_latency - mt_np_latency)/mt_p_latency

                                if p_vs_np_percent_latency_reduction > 50:
                                    print(row)
                                 
                                if max_p_vs_np_entry is None:
                                    max_p_vs_np_entry = {
                                        
                                        "row": row
                                    }
                                else:
                                    if max_p_vs_np_entry[""]
                                
                                mt_np_percent_reduction_array.append(percent_latency_reduction)
                                p_vs_np_percent_reduction_array.append(p_vs_np_percent_latency_reduction)
                                count_dict["mt_np"] += 1
                        else:
                            count_dict["st"] += 1
                
                mt_p_percent_reduction_array = np.array(mt_p_percent_reduction_array, dtype=float)
                mt_np_percent_reduction_array = np.array(mt_np_percent_reduction_array, dtype=float)
                p_vs_np_percent_reduction_array = np.array(p_vs_np_percent_reduction_array, dtype=float)

                mean_mt_p = mt_p_percent_reduction_array.mean()
                mean_mt_np = mt_np_percent_reduction_array.mean()
                mean_p_vs_np = p_vs_np_percent_reduction_array.mean()

                max_mt_p = mt_p_percent_reduction_array.max()
                max_mt_np = mt_np_percent_reduction_array.max()
                max_p_vs_np = p_vs_np_percent_reduction_array.max()

                total_eval = count_dict["mt_p"] + count_dict["mt_np"] + count_dict["st"]
                mt_p_percent = 100*count_dict["mt_p"]/total_eval
                mt_np_percent = 100*count_dict["mt_np"]/total_eval

                mean_lat = total_lat/total_req
                st_mean_lat = total_st_lat/total_req 

                percent_diff = abs(100*(mean_lat - st_mean_lat)/st_mean_lat)

                """
                    Output format: mt_label, mean-p, mean-np, max-p, max-np 
                """
                output_array = [mt_label, mean_mt_p, mean_mt_np, max_mt_p, max_mt_np, mt_p_percent, 
                                mt_np_percent, mean_lat, st_mean_lat, percent_diff, mean_p_vs_np, max_p_vs_np]
                str_to_write = ",".join([str(_) for _ in output_array])
                print(str_to_write)
                out_file = pathlib.Path("{}_best_mt_cache.csv".format(write_policy))
                with open(out_file, "a+") as f:
                    f.write("{}\n".format(str_to_write))


if __name__ == "__main__":
    per_mt_cache = PerMTCache()
    per_mt_cache.run()
    
    
