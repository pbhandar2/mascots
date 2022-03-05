from stat_max_hmrc import StatMaxHMRC
import argparse, json 

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

    

    plotter = StatMaxHMRC(args.mt_label, args.alg, args.step_size)
    for write_policy in ["wb", "wt"]:
        plotter.load_data(write_policy, args.workload_name)
        plotter.plot_num_points_vs_error(write_policy, args.workload_name)