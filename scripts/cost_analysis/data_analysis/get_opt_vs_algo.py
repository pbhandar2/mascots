import argparse, json, pathlib 
import pandas as pd 

from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

OPT_VS_ALGO_HEADER_LIST = ["cost", "opt_type", "opt_t1", "opt_t2", "opt_lat", "hmrc_type", "hmrc_t1", "hmrc_t2", "hmrc_lat", "error"]


class HMRCAnalysis:
    def __init__(self):
        self.experiment_config_file = "../../experiment_config.json"
        self.alg = "v2"

        with open(self.experiment_config_file) as f:
            self.config = json.load(f)

        self.hmrc_dir = pathlib.Path(self.config["mhmrc_cost_analysis_dir"])
        self.ex_dir = pathlib.Path(self.config["ex_cost_analysis_dir"])
        self.out_dir = pathlib.Path(self.config["error_per_mt_config"])
        self.out_dir.mkdir(exist_ok=True)

    
    def get_opt_type_and_latency(self, opt_row):
        st_latency = opt_row["st_latency"]
        mt_p_latency = opt_row["mt_p_latency"]
        mt_np_latency = opt_row["mt_np_latency"]

        if st_latency <= min(mt_p_latency, mt_np_latency):
            t1_size = opt_row["st_size"]
            t2_size = 0 
            return 0, st_latency, [t1_size, 0]
        else:
            if mt_p_latency < mt_np_latency:
                size_str = opt_row["mt_p_size_array"]
                size_array = [int(_) for _ in size_str.split("-")]
                return 1, mt_p_latency, size_array
            else:
                size_str = opt_row["mt_np_size_array"]
                size_array = [int(_) for _ in size_str.split("-")]
                return 2, mt_np_latency, size_array

    
    def get_hmrc_opt_type_and_latency(self, hmrc_row):
        if hmrc_row["latency"] < hmrc_row["st_latency"]:
            return 1, hmrc_row["latency"], [hmrc_row["t1_size"], hmrc_row["t2_size"]]
        else:
            assert(hmrc_row["t2_size"] == 0)
            return 0, hmrc_row["st_latency"] , [hmrc_row["t1_size"], hmrc_row["t2_size"]]

    
    def generate_error_all_mt_labels(self, write_policy, step_size, num_points):
        for mt_label in self.config["priority_2_tier_mt_label_list"]:
            print("evaluating {} wp {} step size {} num points {}".format(mt_label, write_policy, step_size, num_points))
            self.generate_error(mt_label, write_policy, step_size, num_points)


    def generate_error(self, mt_label, write_policy, step_size, num_points):
        out_dir = self.out_dir.joinpath(mt_label, write_policy, str(step_size), str(num_points))
        out_dir.mkdir(exist_ok=True, parents=True)

        for workload_dir in self.hmrc_dir.joinpath(mt_label, write_policy).iterdir():
            workload_name = workload_dir.name 
            hmrc_data_path = workload_dir.joinpath(self.alg, str(step_size), "{}.csv".format(str(num_points)))
            ex_data_path = self.ex_dir.joinpath(mt_label, write_policy, "{}.csv".format(workload_name))

            if not ex_data_path.exists():
                continue 

            if not hmrc_data_path.exists():
                continue 

            hmrc_df = pd.read_csv(hmrc_data_path, names=MHMRC_ROW_JSON)
            hmrc_df.astype({
                "cost": int, 
                "st_latency": float, 
                "t1_size": int, 
                "t2_size": int,
                "latency": float
            })
            ex_df = pd.read_csv(ex_data_path, names=OPT_ROW_JSON)
            ex_df.astype({
                "cost": int, 
                "st_size": int, 
                "st_latency": float, 
                "mt_p_latency": float,
                "mt_np_latency": float
            })

            out_file_path = out_dir.joinpath("{}.csv".format(workload_name))
            f = open(out_file_path, "w+")
            error_array = []
            for _, row in ex_df.iterrows():
                cost_value = row["cost"]
                hmrc_row = hmrc_df[hmrc_df["cost"]==cost_value]

                if len(hmrc_row) == 0:
                    continue 

                hmrc_row = hmrc_row.iloc[0]
                opt_type, opt_latency, opt_size_array = self.get_opt_type_and_latency(row)
                hmrc_opt_type, hmrc_latency, hmrc_size_array = self.get_hmrc_opt_type_and_latency(hmrc_row)
                percent_error = 100*(hmrc_latency - opt_latency)/opt_latency

                out_array = [cost_value, opt_type, 
                        opt_size_array[0], opt_size_array[1], opt_latency, hmrc_opt_type,
                        hmrc_size_array[0], hmrc_size_array[1], hmrc_latency, percent_error]
                f.write("{}\n".format(",".join([str(_) for _ in out_array])))
                error_array.append(percent_error)
            f.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate error data from HMRC and OPT data.")
    parser.add_argument("--workload_name",
        help="Label of the workload")
    parser.add_argument("--mt_label",
        default="D1_S1_H1",
        help="The label of the MT cache")
    parser.add_argument("--wp",
        default="wb",
        help="Write policy")
    parser.add_argument("--num_points", 
        default=1,
        type=int,
        help="Number of points evaluted")
    parser.add_argument("--step_size", 
        default=1,
        type=int,
        help="Step size for analysis")
    args = parser.parse_args()

    for wp in ["wb", "wt"]:
        analysis = HMRCAnalysis()
        analysis.generate_error_all_mt_labels(wp, args.step_size, args.num_points) 
        # analysis.generate_error(args.mt_label, wp, args.step_size, args.num_points) 