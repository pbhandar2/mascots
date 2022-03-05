import numpy as np 
import argparse 
import pathlib 
import json 
import math 

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler

EXCLUSIVE_FLAG = "exclusive"
INCLUSIVE_FLAG = "inclusive"
RD_HIST_DIR = pathlib.Path("/research2/mtc/cp_traces/rd_hist_4k")
EX_MAX_HMRC_DIR = pathlib.Path("/research2/mtc/cp_traces/exclusive_max_hmrc_curve_4k")


class HMRC:
    def __init__(self, hmrc_file_path, rd_hist_path, output_dir):
        self.file_path = hmrc_file_path
        self.rd_hist_path = rd_hist_path
        self.workload_name = pathlib.Path(rd_hist_path).stem
        self.profiler = RDHistProfiler(rd_hist_path)
        self.page_size = 4096
        self.output_dir = pathlib.Path(output_dir)
        self.mhrc_wb_output_file = self.output_dir.joinpath("mhrc_wb.csv")
        self.mhrc_wt_output_file = self.output_dir.joinpath("mhrc_wt.csv")
        self.ex_wb_output_file = self.output_dir.joinpath("ex_wb.csv")
        self.ex_wt_output_file = self.output_dir.joinpath("ex_wt.csv")
        self.mhrc_wb_handle = self.mhrc_wb_output_file.open("w+")
        self.mhrc_wt_handle = self.mhrc_wt_output_file.open("w+")
        self.ex_wb_handle = self.ex_wb_output_file.open("w+")
        self.ex_wt_handle = self.ex_wt_output_file.open("w+")

    
    def get_overhead_gain_ratio(self, mt_config, adm_policy=EXCLUSIVE_FLAG):
        # compute the overhead-gain ratio from the device config 
        if adm_policy == EXCLUSIVE_FLAG:
            overhead = mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
            gain = mt_config[2]["read_lat"] - mt_config[1]["read_lat"] - mt_config[0]["read_lat"] \
                - mt_config[1]["write_lat"]
            og_ratio = overhead/gain 
        elif adm_policy == INCLUSIVE_FLAG:
            overhead = mt_config[1]["write_lat"]
            gain = mt_config[2]["read_lat"] - mt_config[1]["read_lat"] - mt_config[1]["write_lat"]
            og_ratio = overhead/gain 
        else:
            raise ValueError("{} not an accepted admission policy. exclusive or inclusive".format(adm_policy))
        return og_ratio


    def get_opt(self, 
        cache_unit_size, 
        mt_config, 
        adm_policy=EXCLUSIVE_FLAG):
        """ Get OPT MT cache configuration 
        
        Parameters
        ----------
        cache_unit_size : int
            the size of the cache based on the unit of the RD histogram 
        mt_config : list
            list of dicts each representing a cache device at a tier 
        adm_policy : str 
            admission policy (exclusive and inclusive)
        """
        og_ratio = self.get_overhead_gain_ratio(mt_config, adm_policy=adm_policy)

        # find the cost per page and per allocation unit 
        cost_per_page_array = np.zeros(len(mt_config))
        cost_per_unit = np.zeros(len(mt_config))
        for i, device_config in enumerate(mt_config):
            cost_per_page_array[i] = self.page_size * (device_config["cost"]/(device_config["size"]*1e9))
            cost_per_unit[i] = cost_per_page_array[i] * cache_unit_size

        # get the latency of WB and WT MT cache 
        wb_lat_array = self.profiler.two_tier_exclusive_wb(mt_config)
        wt_lat_array = self.profiler.two_tier_exclusive_wt(mt_config)
        
        # get the max cost for this workload 
        hmrc = np.genfromtxt(self.file_path, delimiter="\n")
        min_cost = 5
        max_cost = max(math.ceil(len(hmrc)*cost_per_page_array[0]), min_cost)

        # evaluate each cost value 
        for cur_cost in range(min_cost, max_cost+1, min_cost):
            max_t1_size_pages = math.floor(cur_cost/cost_per_page_array[0])

            # find index of the array where HMRC is greater than OG ratio 
            cur_hmrc = hmrc[:max_t1_size_pages]
            index_array = np.argwhere(cur_hmrc>og_ratio)
            
            filter_points = len(index_array)
            percentage_points_reduced = filter_points/max_t1_size_pages

            # setup the output data and file 
            min_lat_entry = {
                "cost": cur_cost,
                "og_ratio": og_ratio,
                "st_size": 0,
                "st_lat": 0,
                "st_lat_per_dollar": 0,
                "t1_size": 0,
                "t2_size": 0,
                "mt_lat": math.inf,
                "mt_lat_per_dollar": 0.0,
                "max_points": len(hmrc),
                "filtered_points": filter_points,
                "percent_eval": percentage_points_reduced,
                "st_mt_lat_percent_diff": 0.0
            }
            wb_min_lat_entry = min_lat_entry.copy()
            wt_min_lat_entry = min_lat_entry.copy()
            max_t1_size_unit = math.floor(cur_cost/cost_per_unit[0])
            wt_min_lat_entry["st"] = max_t1_size_unit 
            wb_min_lat_entry["st"] = max_t1_size_unit 
            wt_min_lat_entry["st_lat"] = self.profiler.get_mean_latency(max_t1_size_unit * cache_unit_size,
                0, wt_lat_array)
            wb_min_lat_entry["st_lat"] = self.profiler.get_mean_latency(max_t1_size_unit * cache_unit_size,
                0, wb_lat_array)

            for t1_size_unit in range(1, max_t1_size_unit+1):

                # compute the cost left for T2 size and its size 
                cur_t1_cost = t1_size_unit * cost_per_unit[0]
                t2_cost = cur_cost - cur_t1_cost
                t2_size_unit = math.floor(t2_cost/cost_per_unit[1])

                if t2_size_unit < 1:
                    continue 

                mean_lat_wb = self.profiler.get_mean_latency(t1_size_unit*cache_unit_size,
                    t2_size_unit*cache_unit_size, wb_lat_array)

                mean_lat_wt = self.profiler.get_mean_latency(t1_size_unit*cache_unit_size,
                    t2_size_unit*cache_unit_size, wt_lat_array)

                if mean_lat_wb < wb_min_lat_entry["lat"]:
                    wb_min_lat_entry["t1"] = t1_size_unit
                    wb_min_lat_entry["t2"] = t2_size_unit
                    wb_min_lat_entry["lat"] = mean_lat_wb 

                if mean_lat_wt < wt_min_lat_entry["lat"]:
                    wt_min_lat_entry["t1"] = t1_size_unit
                    wt_min_lat_entry["t2"] = t2_size_unit
                    wt_min_lat_entry["lat"] = mean_lat_wt 

            """ Write WT and WB output to each file for this technique """
            mhrc_wb_output_string = ",".join("{}".format(e) for e in wb_min_lat_entry.values())
            print("MHRC WB: {}".format(mhrc_wb_output_string))
            self.mhrc_wb_handle.write("{}\n".format(mhrc_wb_output_string))

            mhrc_wt_output_string = ",".join("{}".format(e) for e in wt_min_lat_entry.values())
            print("MHRC WT: {}".format(mhrc_wt_output_string))
            self.mhrc_wt_handle.write("{}\n".format(mhrc_wt_output_string))

            """ Write WT and WB output to each file using exhaustive search """
            exhaustive_wt_entry = self.profiler.cost_eval_exclusive(mt_config, "wt", cur_cost, cache_unit_size,
                min_t2_size=1)
            exhaustive_wb_entry = self.profiler.cost_eval_exclusive(mt_config, "wb", cur_cost, cache_unit_size,
                min_t2_size=1)

            ex_wb_output_string = ",".join("{}".format(e) for e in exhaustive_wb_entry.values())
            print("EX WB: {}".format(ex_wb_output_string))
            self.ex_wb_handle.write("{}\n".format(ex_wb_output_string))

            ex_wt_output_string = ",".join("{}".format(e) for e in exhaustive_wt_entry.values())
            print("EX WT: {}".format(ex_wt_output_string))
            self.ex_wt_handle.write("{}\n".format(ex_wt_output_string))

            # log the difference from exhaustive to this technique 
            mhrc_min_lat_wb = min(wb_min_lat_entry["st_lat"], wb_min_lat_entry["lat"])
            ex_min_lat_wb = min(exhaustive_wb_entry["st_lat"], 
                min(exhaustive_wb_entry["mt_np_lat"], exhaustive_wb_entry["mt_p_lat"]))
            mhrc_min_lat_wt = min(wt_min_lat_entry["st_lat"], wt_min_lat_entry["lat"])
            ex_min_lat_wt = min(exhaustive_wt_entry["st_lat"], 
                min(exhaustive_wt_entry["mt_np_lat"], exhaustive_wt_entry["mt_p_lat"]))

            wb_diff = (mhrc_min_lat_wb - ex_min_lat_wb)/ex_min_lat_wb
            wt_diff = (mhrc_min_lat_wt - ex_min_lat_wt)/ex_min_lat_wt
            print("WB: {} WT: {}".format(wb_diff, wt_diff))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute OPT MT cache from HMRC")
    parser.add_argument("workload", type=str,
        help="workload_name")
    args = parser.parse_args()

    rd_hist_path = RD_HIST_DIR.joinpath("{}.csv".format(args.workload))
    max_hmrc_path = EX_MAX_HMRC_DIR.joinpath("{}.csv".format(args.workload))

    device_data = {}
    device_config_file = "sb_device_config.json"
    with open(device_config_file) as f:
        device_data = json.load(f)

    mhrc_opt_dir = pathlib.Path("/research2/mtc/cp_traces/mhrc_opt/{}".format(args.workload))
    mhrc_opt_dir.mkdir(parents=True, exist_ok=True)

    hmrc = HMRC(max_hmrc_path, rd_hist_path, mhrc_opt_dir)   
    hmrc.get_opt(256, device_data)