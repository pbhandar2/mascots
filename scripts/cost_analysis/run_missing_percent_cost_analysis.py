import argparse, itertools, time, json, pathlib, math 
import pandas as pd 
import numpy as np 
import multiprocessing as mp

from CostAnalysis import CostAnalysis
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON


def run_cost_analysis(workload_name, cost):
    print("workload {}, cost {}".format(workload_name, cost))
    analysis = CostAnalysis(workload_name, cost)
    analysis.run()


def get_missing_cost_array(workload_name, num_points, mt_label="D1_S1_H1"):
    with open("../experiment_config.json") as f:
        exp_config = json.load(f)

    max_cost_df = pd.read_csv("../workload_analysis/max_cost.csv", 
                                delimiter=" ",
                                names=["workload", "max_cost"])

    ex_out_dir = pathlib.Path(exp_config["ex_cost_analysis_dir"])
    ex_out_path = ex_out_dir.joinpath(mt_label, "wb", "{}.csv".format(workload_name))

    evaluated_cost_array = np.array([], dtype=int)
    if ex_out_path.exists():
        df = pd.read_csv(ex_out_path, names=OPT_ROW_JSON)
        evaluated_cost_array = df["cost"].to_numpy()

    max_cost = max_cost_df[max_cost_df["workload"]==workload_name].iloc[0]["max_cost"]

    all_cost_array = []
    for i in range(10, 101, 10):
        all_cost_array.append(math.ceil(max_cost*i/100))
    remaining_cost_array = np.setdiff1d(all_cost_array, evaluated_cost_array)

    if len(remaining_cost_array) > 0:
        return list(set(remaining_cost_array[:num_points]))
    else:
        return []


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                description="Spawn processing for each cost value for a given workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("num_points", type=int, help="Number of points to evaluate")
    args = parser.parse_args()

    start = time.perf_counter()
    cost_array = get_missing_cost_array(args.workload_name, args.num_points)

    if len(cost_array) > 0:
        print("Evaluating cost ", cost_array)
        arg_list = list(itertools.product([args.workload_name], cost_array))
        with mp.Pool(processes=args.num_points) as pool:
            results = pool.starmap(run_cost_analysis, arg_list)

    end = time.perf_counter()
    print("Workload {}, Time {}".format(args.workload_name, end-start))