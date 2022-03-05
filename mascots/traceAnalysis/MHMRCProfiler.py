import math, json, copy, time
import numpy as np 

import logging
logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib

EXCLUSIVE_FLAG = "exclusive"
INCLUSIVE_FLAG = "inclusive"

MHMRC_ROW_JSON = {
    "cost": 0.0, # max cost of MT cache in dollars 
    "og_ratio": 0.0, # overhead-gain ratio of the MT cache 
    "max_points": 0, # maximum number of points to be evaluated 
    "num_eval": 0, # number of evaluations 
    "st_size": 0, # size of ST cache for a given cost 
    "st_latency": 0, # latency of the ST cache 
    "num_points_filtered": 0, # points filtered using OG ratio 
    "percent_points_filtered": 0.0, # percentage of points filtered using OG ratio 
    "t1_size": 0, # the t1 size found based on MHMRC
    "t2_size": 0, # the t2 size found based on MHMRC
    "latency": 0.0, # the mean latency of the config found by MHMRC
    "cache_cost": 0.0, # actual cost of the cache 
    "mt_label": "",
    "mt_opt_flag": 0 # flag denoting if MT cache is more optimal than ST cache 
}


class MHMRCProfiler:
    """ Max Hit-Miss Ratio Curve (MHMRC) profiler. It generates cost optimal cache for 
        a given workload, MT cache configuration and cost using MHMRC. 

    Attributes
    ----------
    rd_hist_path : str
        path to the file with reuse distance histogram 
    mhmrc_path : str
        path to the file with max hit-miss ratio curve 
    cache_unit_size : int 
        unit of cache allocation in pages 
    page_size : int 
        page size in bytes (optional) (Default: 4096)
    rd_hist_profiler : RDHistProfiler 
        reuse distance histogram profiler to evaluate caches 
    mhmrc : np.array 
        np array containing the max hit-miss ratio curve 
    """

    def __init__(self, rd_hist_path, 
        page_size=4096):
        """
        Parameters
        ----------
        rd_hist_path : str
            path to the file with the RD histogram 
        mhrmc_path : str 
            path to the file with the max hit-miss ratio curve 
        cache_unit_size : int 
            the allocation size of cache in pages 
        page_size : int 
            size of a page in bytes (optional) (Default: 4096)
        """

        self.rd_hist_path = rd_hist_path 
        self.mt_config_lib = MTConfigLib()
        self.rd_hist_profiler = RDHistProfiler(rd_hist_path, page_size=page_size)
        self.mhmrc = self.rd_hist_profiler.max_hit_miss_ratio()


    def get_overhead_gain_ratio(self, mt_config, adm_policy=EXCLUSIVE_FLAG):
        """ Compute the overhead-gain ratio for a given MT configuration. 

        Parameters
        ----------
        mt_config : dict 
            dict containing the cache device specification at each tier 
        adm_policy : str 
            string denoting the admission policy ("exclusive" or "inclusive")

        Returns
        -------
        og_ratio : float 
            overhead-gain ratio of the second tier 
        """

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


    def get_scaled_mhmrc(self, mhmrc, mt_config, cost):
        """ Scale the Max-HMRC based on the size of T2. 

        Parameters
        ----------
        mhmrc : np.array
            array containing Max-HMR at different cache sizes 
        mt_config : json 
            JSON containing device specifications at each tier 
        cost : int 
            the max cost of the cache 

        Return 
        ------
        scaled_mhmrc : np.array 
            Max-HMRC sclaed based on the size of T2 and max possible size of T2 
        """

        scaled_mhmrc = np.copy(mhmrc)
        max_cache_size = self.rd_hist_profiler.max_cache_size
        t1_cost_per_unit = self.mt_config_lib.get_unit_cost(mt_config, 0, 
                                self.rd_hist_profiler.allocation_unit) 
        max_t1_size = len(mhmrc)
        for t1_size in range(1, max_t1_size+1):
            t1_cost = t1_size * t1_cost_per_unit

            if t1_cost < cost:
                t2_cost = cost - t1_cost 
                t2_size = self.mt_config_lib.get_max_tier_size(mt_config, 1, 
                                self.rd_hist_profiler.allocation_unit, t2_cost)

                """ If the sum of sizes of tier 1 and tier 2 is greater or 
                    equal to the working set size then Hit-Miss Ratio does 
                    not need to be scaled. 
                """
                if t1_size + t2_size < max_cache_size:
                    t2_hit_rate = self.rd_hist_profiler.mrc[t1_size+t2_size] - self.rd_hist_profiler.mrc[t1_size]
                    max_t2_hit_rate = self.rd_hist_profiler.mrc[-1] - self.rd_hist_profiler.mrc[t1_size]
                    scaled_mhmrc[t1_size-1] *= (t2_hit_rate/max_t2_hit_rate)
            else:
                scaled_mhmrc[t1_size-1] = 0
        return scaled_mhmrc


    def init_output_json(self, mt_config, cost, num_points_filtered, write_policy="wb"):
        """ Initiate an output based on a template. 

        Parameters
        ----------
        cost : int 
            max cost of the cache 
        mt_config : dict 
            dict containing device specification of each tier 
        cost : int
            max cost of cache in dollars 
        num_points_filtered : int 
            number of points filtered using OG ratio 
        write_policy : str 
            write policy of the cache (optional) (Default: "wb")

        Return 
        ------
        out_json : json
            JSON containing the base information about the output 
        """

        out_json = MHMRC_ROW_JSON.copy()
        out_json["cost"] = cost 
        out_json["og_ratio"] = self.get_overhead_gain_ratio(mt_config) 
        out_json["num_points_filtered"] = num_points_filtered

        if write_policy == "wb":
            lat_array = self.mt_config_lib.two_tier_exclusive_wb(mt_config)
        elif write_policy == "wt":
            lat_array = self.mt_config_lib.two_tier_exclusive_wt(mt_config)
        else:
            raise ValueError("Invalid write policy: {}".format(write_policy))

        max_t1_size = self.mt_config_lib.get_max_tier_size(mt_config, 0, 
                        self.rd_hist_profiler.allocation_unit, cost)
        out_json["st_size"] = max_t1_size
        out_json["st_latency"] = self.rd_hist_profiler.get_mean_latency([max_t1_size, 0], lat_array)
        out_json["max_points"] = max_t1_size
        out_json["percent_points_filtered"] = 100*num_points_filtered/max_t1_size
        out_json["t1_size"] = max_t1_size
        out_json["t2_size"] = 0
        out_json["latency"] = self.rd_hist_profiler.get_mean_latency([max_t1_size, 0], lat_array)
        out_json["mt_label"] = self.mt_config_lib.get_mt_label(mt_config)
        
        return out_json 
        
        
    def eval_mhmrc(self, mhmrc, mt_config, cost,
        step_size = 1, 
        eval_record_points=[]):
        """ Find OPT cache from a MHMRC, mt_config and cost. 

        Parameters
        ----------
        mhmrc : np.array 
            Max Hit-Miss Ratio values for different tier 1 sizes 
        mt_config : dict 
            a dict containing device specifications of each tier 
        cost : int 
            cost of cache in dollars 

        Return
        ------
        wb_out_json : dict 
            output dict for write-back cache 
        wt_out_json : dict 
            output dict for write-through cache 
        """

        # indexes that would sort the MHMRC in descending order 
        sorted_mhmrc_index = np.flip(mhmrc.argsort())
        sorted_mhmrc = mhmrc[sorted_mhmrc_index]

        # filter the MHMRC based on OG ratio 
        filtered_mhmrc = np.where(sorted_mhmrc>self.get_overhead_gain_ratio(mt_config))[0]

        # number of points filtered using the OG ratio 
        num_points_filtered = len(sorted_mhmrc) - len(filtered_mhmrc)

        print(mhmrc[:5])
        print(len(sorted_mhmrc))
        print(self.mt_config_lib.get_max_tier_size(mt_config, 0, 
            self.rd_hist_profiler.allocation_unit, cost))

        """ wb_out_json and wt_out_json contain the best write-back and 
            write-through MT cache respectively identified when different 
            number of points have been evaluated. The key to these dicts 
            are integers that represent the number of points evaluated and
            the value is another dict containing information regarding 
            the MT cache generated. 

            The key 0 contains the MT cache generated when all the points
            are evaluated. 
        """
        wb_out_json, wt_out_json = {}, {}
        wb_out_json[0] = self.init_output_json(mt_config, cost, 
            num_points_filtered, write_policy="wb")
        wt_out_json[0] = self.init_output_json(mt_config, cost, 
            num_points_filtered, write_policy="wt")

        wb_lat_array = self.mt_config_lib.two_tier_exclusive_wb(mt_config)
        wt_lat_array = self.mt_config_lib.two_tier_exclusive_wt(mt_config)

        if len(eval_record_points) == 0:
            eval_record_points = [1,5] + list(range(10, len(filtered_mhmrc), 10))

        num_eval = 0 
        for index in range(0, len(filtered_mhmrc), step_size):
            num_eval += 1

            # cache size represented is index + 1
            t1_size = sorted_mhmrc_index[index] + 1
            t1_cost = t1_size * self.mt_config_lib.get_unit_cost(mt_config, 0, 
                self.rd_hist_profiler.allocation_unit) 
            t2_cost = cost - t1_cost
            t2_size = self.mt_config_lib.get_max_tier_size(mt_config, 1, 
                self.rd_hist_profiler.allocation_unit, t2_cost)
            size_array = [t1_size, t2_size]

            """ When the total cache size is higher than the working set size. We 
                can trade the excess slower cache for the faster cache which will 
                improve performance. 
            """
            if t1_size + t2_size > self.rd_hist_profiler.max_cache_size:
                excess_cache = t1_size + t2_size - self.rd_hist_profiler.max_cache_size
                t1_cost = self.mt_config_lib.get_unit_cost(mt_config, 0, 
                    self.rd_hist_profiler.allocation_unit)
                t2_cost = self.mt_config_lib.get_unit_cost(mt_config, 1, 
                    self.rd_hist_profiler.allocation_unit)
                t1_t2_ratio = math.floor(t1_cost/t2_cost) 

                """ If the tier 2 cache size is excessively large, the total 
                    cache size (t1+t2) is larger than the max cache size. This 
                    means the excess t2 size does not yeild additional hits so 
                    it is a waste of space. Instead, we can utilize the cost of 
                    that excess cache to increase the size of T1, if possible, 
                    which would improve performance. 
                """
                if t2_size > t1_t2_ratio and excess_cache > t1_t2_ratio:
                    # trade away as much of the tier 2 cache for tier 1 cache 
                    while t2_size > 0 and t1_size + t2_size > self.rd_hist_profiler.max_cache_size:
                        t1_size += 1 
                        t2_size -= t1_t2_ratio
                        excess_cache -= t1_t2_ratio - 1
                    
                    # make sure there is no residual cost which can be used to get more caceh 
                    cur_cost = self.mt_config_lib.get_cache_cost(mt_config,
                        [t1_size, t2_size], self.rd_hist_profiler.allocation_unit)
                    residue_cost = cost-cur_cost 
                    while residue_cost >= t2_cost:
                        if residue_cost >= t1_cost:
                            t1_units = math.floor(residue_cost/t1_cost)
                            t1_size += t1_units 
                            residue_cost -= t1_units * t1_cost 
                        else:
                            t2_units = math.floor(residue_cost/t2_cost)
                            t2_size += t2_units
                            residue_cost -= t2_units * t2_cost

            wb_latency = self.rd_hist_profiler.get_mean_latency(size_array, wb_lat_array)
            wt_latency = self.rd_hist_profiler.get_mean_latency(size_array, wt_lat_array)

            if wb_latency < wb_out_json[0]["latency"]:
                wb_out_json[0]["latency"] = wb_latency 
                wb_out_json[0]["t1_size"] = t1_size 
                wb_out_json[0]["t2_size"] = t2_size 

            if wt_latency < wt_out_json[0]["latency"]:
                wt_out_json[0]["latency"] = wt_latency 
                wt_out_json[0]["t1_size"] = t1_size 
                wt_out_json[0]["t2_size"] = t2_size 

            wb_out_json[0]["cache_cost"] = self.mt_config_lib.get_cache_cost(mt_config,
                [wb_out_json[0]["t1_size"], wb_out_json[0]["t2_size"]], 
                self.rd_hist_profiler.allocation_unit)
            wt_out_json[0]["cache_cost"] = self.mt_config_lib.get_cache_cost(mt_config,
                [wt_out_json[0]["t1_size"], wt_out_json[0]["t2_size"]], 
                self.rd_hist_profiler.allocation_unit)
        
            if wb_out_json[0]["st_latency"] > wb_out_json[0]["latency"]:
                if wb_out_json[0]["t1_size"] >= wb_out_json[0]["t2_size"]:
                    wb_out_json[0]["mt_opt_flag"] = 2 
                else:
                    wb_out_json[0]["mt_opt_flag"] = 1

            if wt_out_json[0]["st_latency"] > wt_out_json[0]["latency"]:
                if wt_out_json[0]["t1_size"] >= wt_out_json[0]["t2_size"]:
                    wt_out_json[0]["mt_opt_flag"] = 2 
                else:
                    wt_out_json[0]["mt_opt_flag"] = 1

            wb_out_json[0]["num_eval"] = num_eval
            wt_out_json[0]["num_eval"] = num_eval 
            if num_eval in eval_record_points:
                wb_out_json[num_eval] = copy.deepcopy(wb_out_json[0])
                wt_out_json[num_eval] = copy.deepcopy(wt_out_json[0])

        return wb_out_json, wt_out_json


    def eval_mhmrc_v2(self, mhmrc, mt_config, cost,
        write_policy="wb",
        step_size=1, 
        eval_record_points=[]):
        """ Find OPT cache using Hit-Miss ratio, MT config and cost using 
            the 'v2' algorithm 

        Parameters
        ----------
        mhmrc : np.array 
            Max Hit-Miss Ratio values for different tier 1 sizes 
        mt_config : dict 
            a dict containing device specifications of each tier 
        cost : int 
            cost of cache in dollars 
        write_policy : str 
            write policy ("wb" write-back, "wt" write-through) (Optional) (Default: 'wb')
        step_size : int 
            step size used in the 'v2' algorithm (Optional) (Default: 1)
        eval_record_points : list
            list of eval count at which to record (Optional) (Default: [])

        Return
        ------
        out_json : dict 
            dictionary with 
        """

        # indexes that would sort the MHMRC in descending order 
        sorted_mhmrc_index = np.flip(mhmrc.argsort())
        sorted_mhmrc = mhmrc[sorted_mhmrc_index]

        # filter the MHMRC based on OG ratio 
        filtered_mhmrc = np.where(sorted_mhmrc>self.get_overhead_gain_ratio(mt_config))[0]

        # number of points filtered using the OG ratio 
        num_points_filtered = len(sorted_mhmrc) - len(filtered_mhmrc)

        if len(eval_record_points) == 0:
            eval_record_points = [1,5] + list(range(10, len(filtered_mhmrc), 10))

        """ wb_out_json and wt_out_json contain the best write-back and 
            write-through MT cache respectively identified when different 
            number of points have been evaluated. The key to these dicts 
            are integers that represent the number of points evaluated and
            the value is another dict containing information regarding 
            the MT cache generated. 

            The key 0 contains the MT cache generated when all the points
            are evaluated. 
        """
        out_json = {}
        out_json[0] = self.init_output_json(mt_config, cost, 
                        num_points_filtered, write_policy=write_policy)
        out_json[0]["cache_cost"] = self.mt_config_lib.get_cache_cost(mt_config,
                                        [out_json[0]["st_size"], 0], 
                                        self.rd_hist_profiler.allocation_unit)

        t1_size = sorted_mhmrc_index[0] + 1
        t2_size = self.mt_config_lib.get_max_t2_size(mt_config, t1_size,
                    cost, self.rd_hist_profiler.allocation_unit)

        size_array = [t1_size, t2_size]

        if write_policy == "wb":
            lat_array = self.mt_config_lib.two_tier_exclusive_wb(mt_config)
        else:
            lat_array = self.mt_config_lib.two_tier_exclusive_wt(mt_config)

        mean_latency = self.rd_hist_profiler.get_mean_latency(size_array, lat_array)
        if mean_latency < out_json[0]["st_latency"]:
            front_hmr_index_array = np.where(sorted_mhmrc_index<=sorted_mhmrc_index[0])
            rear_hmr_index_array = np.where(sorted_mhmrc_index>sorted_mhmrc_index[0])
        else:
            front_hmr_index_array = np.where(sorted_mhmrc_index>=sorted_mhmrc_index[0])
            rear_hmr_index_array = np.where(sorted_mhmrc_index<sorted_mhmrc_index[0])

        # hmr_index_array = np.concatenate([front_hmr_index_array[0], rear_hmr_index_array[0]])
        hmr_index_array = front_hmr_index_array[0]

        num_eval = 1 
        if out_json[0]["st_latency"] > mean_latency:
            if out_json[0]["t1_size"] >= out_json[0]["t2_size"]:
                out_json[0]["mt_opt_flag"] = 2 
            else:
                out_json[0]["mt_opt_flag"] = 1
        out_json[0]["num_eval"] = num_eval
        
        if num_eval in eval_record_points:
            out_json[num_eval] = copy.deepcopy(out_json[0])
            out_json[num_eval] = copy.deepcopy(out_json[0])

        for index in range(0, len(hmr_index_array), step_size):
            t1_size = hmr_index_array[index] + 1
            t2_size = self.mt_config_lib.get_max_t2_size(mt_config, t1_size,
                cost, self.rd_hist_profiler.allocation_unit)

            try:
                assert(t2_size>=0)
            except AssertionError:
                print(t1_size)
                print(t2_size)
                print(out_json[0])
                print(sorted_mhmrc_index)
                print(len(sorted_mhmrc_index))
                print("MHR leng: ".format(len(mhmrc)))

            num_eval += 1
            out_json[0]["num_eval"] = num_eval
            size_array = [t1_size, t2_size]
            mean_latency = self.rd_hist_profiler.get_mean_latency(size_array, lat_array)
            
            if mean_latency < out_json[0]["latency"]:
                out_json[0]["latency"] = mean_latency 
                out_json[0]["t1_size"] = t1_size
                out_json[0]["t2_size"] = t2_size
                out_json[0]["cache_cost"] = self.mt_config_lib.get_cache_cost(mt_config,
                                                [out_json[0]["t1_size"], out_json[0]["t2_size"]], 
                                                self.rd_hist_profiler.allocation_unit)

                if out_json[0]["st_latency"] > out_json[0]["latency"]:
                    if out_json[0]["t1_size"] >= out_json[0]["t2_size"]:
                        out_json[0]["mt_opt_flag"] = 2 
                    else:
                        out_json[0]["mt_opt_flag"] = 1
                else:
                    out_json[0]["mt_opt_flag"] = 0

            if out_json[0]["num_eval"] in eval_record_points:
                out_json[num_eval] = copy.deepcopy(out_json[0])

        return out_json 


    def get_opt_form_hmrc(self, mt_config_file, step_size=1, scaled=True):
        """ Cost-optimality analysis for a given workload and a set of MT cache configurations

        Parameters
        ----------
        mt_config_file : str 
            path to the file containing the details of device at each tier 

        Return 
        ------
        wb_opt_dict, wt_opt_dict : dict, dict 
            an iterator that returns a dictionary where they key is the number of points
            evaluated, and the value is the dictionary containing information about 
            the optimal cache configuration found from evaluating said number of points 
        """


        with open(mt_config_file) as f:
            mt_config = json.load(f)

        max_cache_size = self.rd_hist_profiler.max_cache_size
        max_cost = math.ceil(max_cache_size * \
            self.mt_config_lib.get_unit_cost(mt_config, 0, 
                self.rd_hist_profiler.allocation_unit))

        for cost in range(1, max_cost+2):
            start = time.perf_counter()
            max_t1_size = self.mt_config_lib.get_max_tier_size(mt_config, 0, 
                self.rd_hist_profiler.allocation_unit, cost)
            
            if scaled:
                cur_mhmrc = self.get_scaled_mhmrc(self.mhmrc, mt_config, cost)
                cur_mhmrc = cur_mhmrc[:max_t1_size+1]
            else:
                # filter points based on cache size 
                cur_mhmrc = self.mhmrc[:max_t1_size+1]

            wb_opt_entry_dict, wt_opt_entry_dict = self.eval_mhmrc(cur_mhmrc, mt_config, cost, step_size=step_size)

            end = time.perf_counter()
            logging.info("workload: {} cost: {}, {:.2f}% Done!, Time Taken: {}".format(
                self.rd_hist_path.stem,
                cost, 
                100*cost/(max_cost+1),
                end-start))

            yield wb_opt_entry_dict, wt_opt_entry_dict


    def get_opt_form_hmrc_v2(self, mt_config_file, step_size=1, write_policy="wb"):
        """ Find the optimal cache using Hit-Miss Ratio for a set of 
            devices representing each cache tier. 

        Parameters
        ----------
        mt_config_file : str 
            JSON file containing device specification of each tier

        Return 
        ------
        out_dict : dict
            dict with details of the OPT cache for the given cost and workload 
        """

        with open(mt_config_file) as f:
            mt_config = json.load(f)

        max_cache_size = self.rd_hist_profiler.max_cache_size
        max_cost = math.ceil(max_cache_size * \
            self.mt_config_lib.get_unit_cost(mt_config, 0, 
                self.rd_hist_profiler.allocation_unit))

        for cost in range(1, max_cost+2):
            start = time.perf_counter()
            max_t1_size = self.mt_config_lib.get_max_tier_size(mt_config, 0, 
                            self.rd_hist_profiler.allocation_unit, cost)
            
            cur_mhmrc = self.get_scaled_mhmrc(self.mhmrc, mt_config, cost)
            cur_mhmrc = cur_mhmrc[:max_t1_size]

            out_dict = self.eval_mhmrc_v2(cur_mhmrc, mt_config, cost, 
                            write_policy=write_policy,
                            step_size=step_size)

            end = time.perf_counter()
            logging.info("Workload: {}, Cost: {}, {:.2f}% Done!, Time Taken: {}".format(
                self.rd_hist_path.stem,
                cost, 
                100*cost/(max_cost+1),
                end-start))

            yield out_dict