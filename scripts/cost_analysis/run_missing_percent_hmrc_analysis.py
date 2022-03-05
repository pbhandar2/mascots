import argparse, itertools, time, json, pathlib, math 
import pandas as pd 
import numpy as np 
import multiprocessing as mp

from CostAnalysis import CostAnalysis
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON
from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON

from max_hmrc_cost_analysis import MaxHMRCCostExperiment


def run_cost_analysis(workload_name, step_size, cost):
    with open("../experiment_config.json") as f:
        exp_config = json.load(f)
    for mt_label in exp_config["priority_2_tier_mt_label_list"]:
        print("workload {}, mt_label {}, step_size {}, cost {}".format(workload_name, mt_label, step_size, cost))
        alg="v2"
        mt_config_dir = pathlib.Path("/home/pranav/mtc/mt_config/2")
        mt_config_file_path = mt_config_dir.joinpath("{}.json".format(mt_label))

        with open(mt_config_file_path) as f:
            mt_config = json.load(f)

        write_policy_array = ["wt", "wb"]
        for write_policy in write_policy_array:
            analysis = MaxHMRCCostExperiment(workload_name, mt_config_file_path, write_policy, alg, step_size)
            analysis.run_single(mt_config, cost)


def get_missing_cost_array(workload_name, num_points, step_size, mt_label="D1_D3_H3"):
    with open("../experiment_config.json") as f:
        exp_config = json.load(f)

    max_cost_df = pd.read_csv("../workload_analysis/max_cost.csv", 
                                delimiter=" ",
                                names=["workload", "max_cost"])

    ex_out_dir = pathlib.Path(exp_config["ex_cost_analysis_dir"])
    ex_out_path = ex_out_dir.joinpath(mt_label, "wb", "{}.csv".format(workload_name))

    ex_df = pd.read_csv(ex_out_path, names=OPT_ROW_JSON)
    ex_cost_values_evaluated = ex_df["cost"].to_numpy()

    max_cost = max_cost_df[max_cost_df["workload"]==workload_name].iloc[0]["max_cost"]
    all_cost_array = []
    for i in range(10, 101, 10):
        all_cost_array.append(math.ceil(max_cost*i/100))

    hmrc_out_path = pathlib.Path(exp_config["mhmrc_cost_analysis_dir"]).joinpath(mt_label,
                                                                                "wb",
                                                                                workload_name,
                                                                                "v2",
                                                                                str(step_size),
                                                                                "0.csv")

    evaluated_hmrc_cost_array = np.array([], dtype=int)
    if hmrc_out_path.exists():
        hmrc_df = pd.read_csv(hmrc_out_path, names=MHMRC_ROW_JSON)
        evaluated_hmrc_cost_array = hmrc_df["cost"].to_numpy()
    
    remaining_cost_array = np.setdiff1d(ex_cost_values_evaluated, evaluated_hmrc_cost_array)

    if len(remaining_cost_array) > 0:
        return list(set(remaining_cost_array[:num_points]))
    else:
        return []


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                description="Spawn processing for each cost value for a given workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("step_size", type=int, help="Step size of the algorithm")
    parser.add_argument("num_points", type=int, help="Number of points to evaluate")
    args = parser.parse_args()

    start = time.perf_counter()
    cost_array = get_missing_cost_array(args.workload_name, args.num_points, args.step_size)
    print(cost_array)
    if len(cost_array) > 0:
        print("Evaluating cost ", cost_array)
        arg_list = list(itertools.product([args.workload_name], [args.step_size], cost_array))
        with mp.Pool(processes=args.num_points) as pool:
            results = pool.starmap(run_cost_analysis, arg_list)

    end = time.perf_counter()
    print("Workload {}, Time {}".format(args.workload_name, end-start))