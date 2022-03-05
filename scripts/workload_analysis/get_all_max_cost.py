import argparse, json, pathlib, math
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib 

def main(device_label):

    with open("../experiment_config.json") as f:
        experiment_config = json.load(f)
    with open("../max_mhrc/device_list.json") as f:
        device_list = json.load(f)

    device_info = list(filter(lambda d: d["label"] == device_label, device_list))[0]
    mt_config_helpers = MTConfigLib()
    unit_cost = mt_config_helpers.get_unit_cost([device_info], 0, 256)
    
    rd_hist_dir = pathlib.Path(experiment_config["rd_hist_4k_dir"])
    output_file_path = "/research2/mtc/cp_traces/general/max-cost-cp-{}.csv".format(device_label)
    with open(output_file_path, "w+") as f:
        for rd_hist_path in rd_hist_dir.iterdir():
            print("Evaluating {}".format(rd_hist_path))
            profiler = RDHistProfiler(rd_hist_path)
            max_cost = math.ceil(unit_cost * profiler.max_cache_size)
            workload_name = rd_hist_path.stem
            f.write("{},{}\n".format(workload_name, max_cost))
            f.flush()
            print("{},{}\n".format(workload_name, max_cost))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate max cost for each workload given a tier 1 cache device")
    parser.add_argument("adevice_label")
    args = parser.parse_args()
    main(args.device_label)