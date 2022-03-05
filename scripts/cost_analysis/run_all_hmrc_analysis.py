import argparse, itertools, time, pathlib, json 
import multiprocessing as mp

from max_hmrc_cost_analysis import MaxHMRCCostExperiment

def run_cost_analysis(workload_name, mt_label, step_size):
    print("workload {}, mt_label {}, step_size {}".format(workload_name, mt_label, step_size))
    mt_config_dir = pathlib.Path("/home/pranav/mtc/mt_config/2")
    mt_config_file_path = mt_config_dir.joinpath("{}.json".format(mt_label))
    alg="v2"
    write_policy_array = ["wt", "wb"]

    for write_policy in write_policy_array:
        analysis = MaxHMRCCostExperiment(workload_name, mt_config_file_path, write_policy, alg, step_size)
        analysis.run()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                description="Use Hit-Miss Ratio to find a cost-efficient cache for a given workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("step_size", type=int, help="step size of the algorithm")
    args = parser.parse_args()

    start = time.perf_counter()

    with open("../experiment_config.json", "r") as f:
        experiment_config = json.load(f)
    mt_label_list = experiment_config["priority_2_tier_mt_label_list"]
    
    arg_list = list(itertools.product([args.workload_name], mt_label_list, [args.step_size]))
    with mp.Pool(processes=len(mt_label_list)) as pool:
        results = pool.starmap(run_cost_analysis, arg_list)
        
    end = time.perf_counter()
    print("Workload {}, Time {}".format(args.workload_name, end-start))