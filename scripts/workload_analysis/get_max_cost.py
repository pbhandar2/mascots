import argparse, json, pathlib, math
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib 

def main(workload_name, device_label):

    with open("../experiment_config.json") as f:
        experiment_config = json.load(f)
    with open("../max_mhrc/device_list.json") as f:
        device_list = json.load(f)

    device_info = list(filter(lambda d: d["label"] == device_label, device_list))[0]
    mt_config_helpers = MTConfigLib()
    unit_cost = mt_config_helpers.get_unit_cost([device_info], 0, 256)
    
    rd_hist_path = pathlib.Path(experiment_config["rd_hist_4k_dir"]).joinpath("{}.csv".format(workload_name))
    profiler = RDHistProfiler(rd_hist_path)
    max_cost = math.ceil(unit_cost * profiler.max_cache_size)
    print(workload_name, max(max_cost,1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate max cost for each workload given a tier 1 cache device")
    parser.add_argument("workload_name")
    parser.add_argument("device_label")
    args = parser.parse_args()
    main(args.workload_name, args.device_label)