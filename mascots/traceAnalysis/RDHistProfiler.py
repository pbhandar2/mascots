import itertools, pathlib, math, json, time, logging, copy
import numpy as np 
import matplotlib.pyplot as plt

from mascots.traceAnalysis.MTConfigLib import MTConfigLib 
logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

EXCLUSIVE_FLAG = "exclusive"
INCLUSIVE_FLAG = "inclusive"

# template of a row when finding cost-efficient cache for a given workload, MT config and cost 
OPT_ROW_JSON = {
    "cost": 0.0, # cost limit of the cache 
    "max_cost": 0.0, # max cost of cache for this workload and MT cache 
    "num_eval": 0, # number of points evaluated 
    "st_size": 0, # size of the base ST cache (one that uses the tier 1 device) (Default: MB)
    "st_latency": 0.0, # latency of the base ST cache (microseconds)
    "st_cost": 0.0, # cost of the base ST cache 
    "st_bandwidth_per_dollar": 0.0, # latency per dollar of the base ST cache 
    "opt_st_size": 0, # size of the best ST cache (one that might not use the tier 1 device)
    "opt_st_latency": math.inf, # latency of the best ST cache 
    "opt_st_cost": 0.0, # cost of the best ST cache 
    "opt_st_bandwidth_per_dollar": 0.0, # latency per dollar of best ST cache 
    "opt_st_device_label": "", # label of the device used in the best ST cache 
    "mt_p_size_array": None, # array of sizes of the cost-optimal pyramidal MT cache 
    "mt_p_latency": math.inf, # latency of cost-optimal pyramidal MT cache (microseconds)
    "mt_p_cost": 0.0, # exact cost of pyramidal MT cache 
    "mt_p_bandwidth_per_dollar": 0.0, # latency per dollar of cost-optimal pyramidal MT cache 
    "mt_np_size_array": None, # array of sizes of the cost-optimal non-pyramidal MT cache 
    "mt_np_latency": math.inf, # latency of cost-optimal non-pyramidal MT cache (microseconds)
    "mt_np_cost": 0.0, # exact cost of non-pyramidal MT cache 
    "mt_np_bandwidth_per_dollar": 0.0, # latency per dollar of cost-optimal non-pyramidal MT cache 
    "mt_opt_flag": 0, # flag denoting optimal MT cache type (0=ST, 1=pyramidal MT, 2=non-pyramidal MT)
    "mt_label": "" # label to identify the MT configuration evaluated (e.g D1_S1_H2)
}

class RDHistProfiler:
    """ RDHistProfiler class allows user to perform single-tier
    and multi-tier cache analyw10sis from a file containing the 
    reuse distance histogram. 

    Attributes
    ----------
    file : pathlib.Path
        path to the reuse distance histogram file 
    page_size : int 
        page size in bytes 
    allocation_size : int 
        the number of pages in a unit allocation 
    rd_hist : np.array 
        np array containing read/write hit count at each cache size 
    cold_miss : np.array(int)
        np array of size (1,2) containing read/write cold miss counts 
    read_count : int 
        the number of read requests 
    write_count : int 
        the number of write requests 
    req_count : int 
        the number of requests 
    max_cache_size : int 
        size of cache that yields that maximum possible hit rate 
    self.hrc : np.array(float)
        np.array representing the Hit Rate Curve (HRC)
    self.mrc : np.array(float)
        np.array representing the Miss Rate Curve (MRC)
    """

    def __init__(self, rd_hist_file_path, 
            page_size=4096, 
            allocation_unit=256, 
            second_scaler=1e6):  
        """ 
        Parameters
        ----------
        rd_hist_file_path: str
            path to the file containing the RD histogram 
        page_size : int 
            page size in bytes (optional) (Default: 4096)
        allocation_unit : int 
            number of pages in a unit allocation (optional) (Default: 256)
        second_scaler : int 
            value to scale bandwidth values by if latency is in microseconds
            then 1e6 would scale it to seconds (optional) (default: 1e6)
        """

        # load RD histogram from the file and group it based on allocation unit
        self.file = pathlib.Path(rd_hist_file_path)
        self.page_size = page_size 
        self.allocation_unit = allocation_unit
        self.second_scaler = second_scaler
        self.mt_lib = MTConfigLib()
        self.load_rd_hist(rd_hist_file_path)


    def load_rd_hist(self, rd_hist_file_path):
        """ Load the RD histogram in a file and update the stats

        Parameters
        ----------
        rd_hist_file_path : str
            path to the file containing RD histogram
        """

        # each row has read and write count at that RD 
        rd_hist_data = np.loadtxt(rd_hist_file_path, delimiter=",", dtype=int)

        # store cold misses in a separate variable 
        self.cold_miss = np.copy(rd_hist_data[0]) 

        # length of RD histogram based on allocation unit 
        len_rd_hist = math.ceil(len(rd_hist_data)/self.allocation_unit)
        self.rd_hist = np.zeros((len_rd_hist+1, 2), dtype=int)
        self.rd_hist[1:] = np.add.reduceat(rd_hist_data[1:], 
            range(0, len(rd_hist_data), self.allocation_unit))

        assert(self.rd_hist[:, 0].sum() == rd_hist_data[1:, 0].sum())
        assert(self.rd_hist[:, 1].sum() == rd_hist_data[1:, 1].sum())

        # track the total read and write count 
        self.read_count = self.rd_hist[:, 0].sum() + self.cold_miss[0]
        self.write_count = self.rd_hist[:, 1].sum() + self.cold_miss[1]
        self.req_count = self.read_count + self.write_count 

        # index of the array corresponds to cache size at which cache hits occur
        # index 0 always has value 0,0 because cache size of 0 yields no hits 
        self.max_cache_size = len(self.rd_hist) - 1 

        # generate MRC/HRC from the RD histogram 
        self.hrc = self.rd_hist.sum(axis=1).cumsum()/self.req_count
        self.mrc = 1 - self.hrc 
        

    def get_exclusive_cache_hits(self, size_array):
        """ Get the number of cache hits at each tier and cache misses. 

        Parameters
        ----------
        size_array : list 
            list of size N cotaning the size of each tier of an N tier cache 

        Return
        ------
        hit_array : np.array
            array of size N+1 with cache hits at each tier of an N tier cache and cache misses 
        """

        # tracker starts from 1 because at index 0 we store cold miss
        rd_hist_index_tracker = 1

        # one additional row for cache misses 
        cache_hits_array = np.zeros((len(size_array)+1, 2), dtype=int)
        for tier_index in range(len(size_array)):

            # if the total cache size exceeds the max cache size we can break additional tier would have 0 hits 
            if size_array[tier_index] + rd_hist_index_tracker >= self.max_cache_size:
                cache_hits_array[tier_index] = self.rd_hist[rd_hist_index_tracker:, :].sum(axis=0)

                # misses are only cold misses since max cache size is acheived 
                cache_hits_array[len(size_array)] = self.cold_miss 
                break 
            else:
                cache_hits_array[tier_index] = self.rd_hist[rd_hist_index_tracker:rd_hist_index_tracker+size_array[tier_index], :].sum(axis=0)
            rd_hist_index_tracker += size_array[tier_index] 
        else:
            # we exit before breaking meaning we have more misses apart from the cold misses 
            cache_hits_array[len(size_array)] = np.sum([self.cold_miss, self.rd_hist[rd_hist_index_tracker:, :].sum(axis=0)], axis=0)

        assert(np.sum(cache_hits_array)==np.sum(self.rd_hist)+np.sum(self.cold_miss))
        return cache_hits_array


    def get_mean_latency(self, size_array, lat_array):
        """ Compute mean latency for a given T1, T2 size and latency array.

        Parameters
        ----------
        size_array : list 
            list of size N cotaning the size of each tier of an N tier cache 
        lat_array : np.array
            array of size (N+1, 2) with read and write latency of each tier and storage 

        Return
        ------
        mean_latency : float
            mean latency of an MT cache 
        """

        cache_hits = self.get_exclusive_cache_hits(size_array)
        total_latency = np.sum(np.multiply(cache_hits, lat_array))
        return total_latency/self.req_count


    def exhaustive_cost_analysis(self, mt_config_file_list, cache_unit_size, 
        min_cost=1, max_cost=-1):
        """ Do an exhaustive cost analysis for this workload, and a list of MT caches 

        Parameters
        ----------
        mt_config_file_list : list 
            list of string representing paths to MT config files 
        cache_unit_size : int 
            the number of pages per allocation unit 
        min_cost : int 
            the minimum cost to evaluate (optional) (Default: 5)
        max_cost : int 
            the maximum cost to evaluate (optional) (Default: -1)

        Return 
        ------
        opt_entry : iterator 
            an iterator that returns the write-back, write-through OPT per iteration 
            using iterator instead of returning an array so that we can write to file 
            as we get results and analyze
        """

        header_dict = {
            "workload": self.file.stem, 
            "cache_unit_size": cache_unit_size, 
            "page_size": self.page_size,
            "max_cost": max_cost }
        logging.info("Performing exhaustive cost analysis")
        for key in header_dict:
            logging.info("{}: {}".format(key, header_dict[key]))

        for mt_config_file in mt_config_file_list:
            mt_config = {}
            with open(mt_config_file) as f:
                mt_config = json.load(f)
            if max_cost == -1:
                tier1_unit_cost = self.mt_lib.get_unit_cost(mt_config, 0, cache_unit_size)
                max_cost = max(math.ceil(self.max_cache_size * tier1_unit_cost), min_cost)
            logging.info("Computing cost range {} to {}".format(min_cost, max_cost))
            for cost in range(min_cost, max_cost+1):
                start = time.perf_counter()
                wb_opt_entry, wt_opt_entry = self.two_tier_optimized_cost_analysis(mt_config, cost)
                end = time.perf_counter()
                logging.info("workload: {} cost: {}, {:.2f}% Done!, Time Taken: {}".format(
                    self.file.stem,
                    cost, 
                    100*cost/max_cost,
                    end-start))
                yield wb_opt_entry, wt_opt_entry


    def get_mean_bandwidth_per_dollar(self, out_dict, cache_type_label):
        """ Compute the mean bandwidth per dollar 

            Parameters
            ----------
            out_dict : dict
                dict containing the latency of the best ST, MT-P and MT-NP cache 
            cache_type_label : str
                label indicating the type of cache 
            Return 
            ------
            mean_bandwidth_per_dollar : float 
                the mean bandwidth per dollar spent 
        """

        bandwidth = self.second_scaler/(out_dict["{}_latency".format(cache_type_label)]*self.allocation_unit)
        return bandwidth/out_dict["{}_cost".format(cache_type_label)]


    def init_two_tier_exhaustive_output(self, mt_config, cost_limit, write_policy):
        """ Generate a base output dict for a given MT configuration, cost limit 
            and write policy. 

            Parameters
            ----------
            mt_config : list 
                list of dicts with device properties 
            cost_limit : int 
                cost limit of MT cache in dollars 
            write_policy : str 
                "wb" (write-back) or "wt" (write-through) write policy 

            Return 
            ------
            output_dict : dict 
                dict containing details of cost analysis 
        """

        mt_lib = MTConfigLib()
        if write_policy == "wb":
            lat_array = mt_lib.two_tier_exclusive_wb(mt_config)
        elif write_policy == "wt":
            lat_array = mt_lib.two_tier_exclusive_wt(mt_config)

        out_dict = copy.deepcopy(OPT_ROW_JSON)
        st_cache_size = mt_lib.get_max_tier_size(mt_config, 0, 
                                                self.allocation_unit, 
                                                cost_limit)
        
        out_dict["cost"] = cost_limit
        out_dict["st_size"] = out_dict["opt_st_size"] = st_cache_size
        out_dict["st_latency"] = out_dict["opt_st_latency"] = self.get_mean_latency([st_cache_size, 0], 
                                                                                    lat_array)
        out_dict["st_cost"] = out_dict["opt_st_cost"] = mt_lib.get_cache_cost(mt_config, 
                                                                                [st_cache_size, 0],
                                                                                self.allocation_unit)
        out_dict["max_cost"] = math.ceil(mt_lib.get_cache_cost(mt_config, [self.max_cache_size, 0],
                                                                self.allocation_unit))
        out_dict["st_bandwidth_per_dollar"] = self.get_mean_bandwidth_per_dollar(out_dict, "st")
        out_dict["opt_st_bandwidth_per_dollar"] = self.get_mean_bandwidth_per_dollar(out_dict, "opt_st")
        out_dict["opt_st_device_label"] = mt_config[0]["label"]
        out_dict["mt_p_size_array"] = [0, 0]
        out_dict["mt_np_size_array"] = [0, 0]
        out_dict["num_eval"] = st_cache_size
        out_dict["mt_label"] = mt_lib.get_mt_label(mt_config)
        return out_dict

    
    def get_opt_cache_type_flag(self, out_dict):
        """ Get the OPT cache type given the output dict containing 
            latency of different cache types 

            Parameters
            ----------
            out_dict : dict
                dict containing the latency of the best ST, MT-P and MT-NP cache 
            Return 
            ------
            opt_cache_type_flag : int 
                flag indicating the type of cache that was optimal 
        """

        st_latency = out_dict["st_latency"]
        mt_p_latency = out_dict["mt_p_latency"]
        mt_np_latency = out_dict["mt_np_latency"]

        mt_opt_flag = 0
        if mt_p_latency < st_latency and mt_p_latency < mt_np_latency:
            mt_opt_flag = 1
        elif mt_np_latency < st_latency and mt_np_latency <= mt_p_latency:
            mt_opt_flag = 2
        return mt_opt_flag 


    def update_opt_cache(self, mt_config, out_dict, size_array, latency, cache_label):
        """ Update the output JSON if necessary

        Parameters
        ----------
        mt_config : list 
            list of dicts with device properties 
        out_dict : dict 
            output dict containing details of OPT cache types 
        size_array : list 
            the size array that was evaluated 
        latency : float 
            the latency of the evaluationn 
        cache_label : str 
            "opt_st" for ST, "mt_p" for MT-P, "mt_np" for MT-NP 
        """
        if latency < out_dict["{}_latency".format(cache_label)]:
            if cache_label == "opt_st":
                out_dict["{}_size".format(cache_label)] = size_array[1]
                out_dict["{}_device_label".format(cache_label)] = mt_config[1]["label"]
                assert(size_array[0] == 0)
            else:
                out_dict["{}_size_array".format(cache_label)] = size_array
            out_dict["{}_latency".format(cache_label)] = latency 
            out_dict["{}_cost".format(cache_label)] = self.mt_lib.get_cache_cost(mt_config, 
                                                                                size_array,
                                                                                self.allocation_unit)
            out_dict["{}_bandwidth_per_dollar".format(cache_label)] = self.get_mean_bandwidth_per_dollar(out_dict,
                                                                                                        cache_label)


    def two_tier_optimized_cost_analysis(self, mt_config, cost_limit):
        """ Find optimal cache given cache devices at each tier and cost limit. 

            Parameters
            ----------
            mt_config : list 
                list of dicts with device specifications 
            cost_limit : int 
                the cost limit of MT cache in dollars 

            Return 
            ------
            wb_out_dict, wt_out_dict : dict 
                output dict with output of exhaustive search for wb and wt 
        """

        wb_output_dict = self.init_two_tier_exhaustive_output(mt_config, cost_limit, "wb")
        wt_output_dict = self.init_two_tier_exhaustive_output(mt_config, cost_limit, "wt")

        wb_lat_array = self.mt_lib.two_tier_exclusive_wb(mt_config)
        wt_lat_array = self.mt_lib.two_tier_exclusive_wt(mt_config)

        t1_cost = self.mt_lib.get_unit_cost(mt_config, 0, self.allocation_unit)
        max_t1_size = math.floor(cost_limit/t1_cost)
        
        for t1_size in range(max_t1_size+1):
            t2_size = self.mt_lib.get_max_t2_size(mt_config, t1_size, cost_limit, 
                                                    self.allocation_unit)
            wb_mean_latency = self.get_mean_latency([t1_size, t2_size], wb_lat_array)
            wt_mean_latency = self.get_mean_latency([t1_size, t2_size], wt_lat_array)
            size_array = [t1_size, t2_size]

            if t1_size == 0:
                self.update_opt_cache(mt_config, wb_output_dict, size_array, wb_mean_latency, "opt_st")
                self.update_opt_cache(mt_config, wt_output_dict, size_array, wt_mean_latency, "opt_st")                
                continue 

            if t1_size < t2_size:
                self.update_opt_cache(mt_config, wb_output_dict, size_array, wb_mean_latency, "mt_p")
                self.update_opt_cache(mt_config, wt_output_dict, size_array, wt_mean_latency, "mt_p")                      
            else:
                self.update_opt_cache(mt_config, wb_output_dict, size_array, wb_mean_latency, "mt_np")
                self.update_opt_cache(mt_config, wt_output_dict, size_array, wt_mean_latency, "mt_np") 

        wb_output_dict["mt_opt_flag"] = self.get_opt_cache_type_flag(wb_output_dict)
        wt_output_dict["mt_opt_flag"] = self.get_opt_cache_type_flag(wt_output_dict)
        
        return wb_output_dict, wt_output_dict


    def max_hit_miss_ratio(self, adm_policy=EXCLUSIVE_FLAG):
        """ Return the Max Hit-Miss Ratio Curve for this RD histogram 

            Parameters
            ----------
            adm_policy : string 
                "exclusive" or "inclusive" admission policy (Optional) 
                    (Default: EXCLUSIVE_FLAG)

            Return
            ------
            max_hmrc_array : np.array 
                np array of size equal to the maximum cache size with 
                    maximum hit-miss ratio at different tier 1 sizes 
        """

        """ Generating an array of read/write hit count
            at different cache sizes. The values at index 
            'i' represents the read/write hit count for 
            a cache of size 'i'. 
            
            For instance,
            At index 0, the values would be [0,0] since a 
            cache of size 0 would yield no read/write hits. 
            
            At index -1, the values would be [x,y] where 
            x and y are maximum possible read and write 
            hit count. """
        cum_hit_count_array = self.rd_hist.cumsum(axis=0)
        max_read_hits = cum_hit_count_array[-1][0]
        max_write_hits = cum_hit_count_array[-1][1]

        """ For every tier 1 size, the array contains the read 
            miss count. The number of read misses at a given 
            tier 1 size is also the potential number of read hits
            for an additional tier. 
            
            NOTE: we ignore index 0 as it means that the value stored 
            is for tier 1 with size 0 which means there is no tier 1. 
            Our metric assumes an existence of a tier 1 cache with size > 0. 
        """
        potential_read_hits_array = max_read_hits - cum_hit_count_array[1:,0]

        if adm_policy == EXCLUSIVE_FLAG:
            """ In an exclusive MT cache with 2 tiers,  
                all read misses incur additional overhead 
                all tier 1 write miss incur additional overhead 
                    - the overhead in the above two cases is due to the 
                        eviction from tier 1 which is to be admitted to 
                        tier 2 """
            miss_array = (max_write_hits + self.cold_miss.sum()) - cum_hit_count_array[1:, 1]
        elif adm_policy == INCLUSIVE_FLAG:
            """ In an inclusve MT cache with 2 tiers, 
                all read misses incur additional overhead 
                all write requests incur additional overhead 
                    - the overhead in the above two cases is due to the 
                        additional write to tier 2 on a tier 2 read 
                        miss and all write requests """
            miss_array = self.cold_miss.sum() + max_write_hits
        else:
            raise ValueError(
                "{} not a valid admission policy. Only exclusive or inclusive".format(adm_policy))
        
        return np.divide(potential_read_hits_array, miss_array)


    def mb_to_bytes(self, mb):
        """ Convert a given value from MB to bytes 

        Parameters
        ----------
        mb : int 
            the value in MB to be converted to bytes
        """

        return mb*1e6


    def get_hmrc(self, filter_size_mb=0):
        """ Get array of Hit-Miss Ratio for each cache size. 

        Parameters
        ----------
        filter_size_mb : int 
            filter removes the RD histogram entries that are 
                not relevant (Default: 0) (optional)

            For instance, if we are generating an MHRC for this 
            RD Histogram and we are evaluating adding a second tier 
            where there the first tier is of size 10MB, we set the 
            filter_size_mb to 10. This filters the requests served 
            by the first tier and we evaluate adding a cache tier 
            based on rest of the request which is tier 1 misses. 
        """

        # filter the requests that are served from the tier 1 cache. 
        filter_size = int(self.mb_to_bytes(filter_size_mb)//self.page_size)
        cum_hit_count_array = self.rd_hist[filter_size+1:].cumsum(axis=0)

        assert(len(cum_hit_count_array) > 0)

        read_count, write_count = cum_hit_count_array[-1][0]+self.cold_miss[0], \
            cum_hit_count_array[-1][1]+self.cold_miss[1]
        miss_count_array = np.zeros(cum_hit_count_array.shape[0], dtype=int) 
        miss_count_array = read_count - cum_hit_count_array[:, 0]
        miss_count_array += write_count

        # read hits whose latency improves 
        hit_array = cum_hit_count_array[:,0] 

        # read miss and write requests where latency gets worse 
        hit_miss_ratio_curve = np.divide(hit_array, miss_count_array)
        return hit_miss_ratio_curve