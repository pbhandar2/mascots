from stat_max_hmrc import StatMaxHMRC
import argparse, json 
import numpy as np 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot number of points against error for a workload.")
    parser.add_argument("workload_name", 
        help="The name of the workload")
    parser.add_argument("--mt_label", 
        default="D1_S1_H1",
        help="The label of MT configuration to evaluate")
    parser.add_argument("--alg", 
        default="v2",
        help="The algorithm to analyze")
    parser.add_argument("--step_size", 
        default=1,
        type=int,
        help="Step size for analysis")
    args = parser.parse_args()

    with open("../../experiment_config.json", "r") as f:
        experiment_config = json.load(f)

    write_policy = "wb"
    for mt_label in experiment_config["priority_2_tier_mt_label_list"]:
        hmrc_analysis = StatMaxHMRC(mt_label, args.alg, args.step_size)
        hmrc_analysis.load_data(args.workload_name, write_policy)
        error = hmrc_analysis.get_error(args.workload_name, write_policy)

        error_array = []
        for cost_value in error:
            data = error[cost_value]
            if 5 not in data:
                continue 
            output_list = data[5]
            if output_list[0] == 0 and output_list[2]> 0:
                error_array.append(output_list[2])
            elif output_list[0] == 1:
                error_array.append(output_list[2])
        
        print(args.workload_name, mt_label, np.array(error_array).mean())
