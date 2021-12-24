import pathlib, math 
import numpy as np 

class RDHistProfiler:
    def __init__(self, rd_hist_file_path):  
        """ RDHistProfiler class allows users to perform analysis on a RD Hist 
        object which represents a workload. 

        Params
        ------
        rd_hist_file_path: path to the RD histogram file (str)
        """

        self.file = pathlib.Path(rd_hist_file_path)

        self.data = np.loadtxt(rd_hist_file_path, delimiter=",", dtype=int)
        self.cold_miss = self.data[0] # record cold misses in a separate variable 
        self.data[0] = np.array([0,0]) # replace cold misses in the data array 

        self.read_count = self.data[:, 0].sum() + self.cold_miss[0]
        self.write_count = self.data[:, 1].sum() + self.cold_miss[1]
        self.req_count = self.read_count + self.write_count 
        self.hrc = self.data/self.req_count
        self.mrc = 1-self.hrc 
        self.max_cache_size = len(self.data) - 1 


    def get_exclusive_cache_hits(self, t1_size, t2_size):
        """ Get the number of cache hits at each tier and the remaining cache misses.

        Params
        ------
        t1_size: size of Tier 1 (int)
        t2_size: size of Tier 2 (int)

        Returns
        -------
        hit_array: array with read and write hits at each tier (np.array)
        """

        if t1_size >= self.max_cache_size:
            t1_hits = self.data[1:,:].sum(axis=0)
            t2_hits = np.array([0,0])
            miss = self.cold_miss
        else:
            t1_hits = self.data[1:t1_size+1, :].sum(axis=0)

            if t1_size + t2_size >= self.max_cache_size:
                t2_hits = self.data[t1_size+1:].sum(axis=0)
                miss = self.cold_miss 
            else:
                t2_hits = self.data[t1_size+1:t1_size+t2_size+1, :].sum(axis=0)
                miss = self.cold_miss + self.data[t1_size+t2_size+1:].sum(axis=0)
        return np.array([t1_hits, t2_hits, miss])


    def get_mean_latency(self, t1_size, t2_size, lat_array):
        """ Compute mean latency for a given T1, T2 size and latency array.

        Params
        ------
        t1_size: size of Tier 1 (int)
        t2_size: size of Tier 2 (int)
        lat_array: array of read and write latency of each tier (np.array) 

        Returns
        -------
        mean_latency: mean latency (float)
        """

        cache_hits = self.get_exclusive_cache_hits(t1_size, t2_size)
        total_latency = np.sum(np.multiply(cache_hits, lat_array))
        return total_latency/self.req_count


    def two_tier_exclusive_wb(self, mt_config):
        """ For a given MT configuration, return the latency of a 
        2-tier write-back exclusive MT cache.

        Params
        ------
        mt_config: JSON object containing the latency and price of the 
            cache device at each cache tier (JSON)

        Returns
        -------
        lat_array: array of read and write latency of each cache tier 
        """

        lat_array = np.zeros([2, len(mt_config)], dtype=float) 

        # Tier 1 Read Lat = read from T1 
        lat_array[0][0] = mt_config[0]["read_lat"]
        # Tier 1 Write Lat = write to T1 
        lat_array[0][1] = mt_config[0]["write_lat"] 

        # Tier 2 Read Lat = read from T2 + write to T1 + read from T1 + write to T2 
        lat_array[1][0] = mt_config[1]["read_lat"] + mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
        # Tier 2 Write Lat = write to T1 + read from T1 + write to T2 
        lat_array[1][1] = self.config[0]["write_lat"] + mt_config[0]["read_lat"] \
            + mt_config[1]["write_lat"] 

        # Read Miss Lat = read from Storage + write to T1 + read from T1 + write to T2 
        lat_array[2][0] = mt_config[-1]["read_lat"] + mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
        # Write Miss Lat = write to T1 + read from T1 + write to T2 
        lat_array[2][1] = mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]

        return lat_array


    def two_tier_exclusive_wt(self, mt_config):
        """ For a given MT configuration, return the latency of a 
        2-tier write-through exclusive MT cache.  

        Params
        ------
        mt_config: JSON object containing the latency and price of the 
            cache device at each cache tier (JSON)

        Returns
        -------
        lat_array: array of read and write latency of each cache tier 
        """

        """ The difference between the latency of an exclusive 2-tier 
        write-back and write-through is that there has to be a disk-write 
        on every write request regardless of the tier of the cache hit. Disk-writes 
        are ignored in write-back caches as it is assumed to be done async so 
        doesn't add to the latency. 
        """

        lat_array = self.two_tier_exclusive_wb(mt_config)
        for i in range(len(mt_config)):
            lat_array[i][1] += self.config[-1]["write_lat"]
        return lat_array


    def cost_eval_exclusive(self, mt_config, write_policy, cost, cache_unit_size):
        """ Evaluate various cache configuration of a given cost and MT cache configuration. 

        Params
        ------
        mt_config: JSON object containing the device properties (JSON)
        write_policy: write policy of the MT cache (str)
        cost: the cost of the cache (float)
        cache_unit_size: the unit size of cache allocation (int)

        Returns
        -------
        mt_cache_info: JSON containing the following keys: 
            cost: cost of the cache 
            st_t1: tier 1 size of ST cache (int)
            st_lat: mean latency of the ST cahce (float)
            mt_p_t1: tier 1 size of pyramidal MT cache (int)
            mt_p_t2: tier 2 size of pyramidal MT cahce (int)
            mt_p_lat: mean latency of pyramidal MT cache (float)
            mt_np_t1: tier 1 size of non-pyramidal MT cache (int)
            mt_np_t2: tier 2 size of non-pyramidal MT cache (int)
            mt_np_lat: mean latency of non-pyramidal MT cache (float)
        """

        if write_policy == "wb":
            lat_array = self.two_tier_exclusive_wb(mt_config)
        elif write_policy == "wt":
            lat_array = self.two_tier_exclusive_wb(mt_config)
        else:
            raise ValueError

        mt_p_config, mt_np_config = [0,0], [0,0]
        mt_p_lat, mt_np_lat, st_lat = math.inf, math.inf, math.inf

        """ For the given cost, find 
            1. The latency and size of an ST cache. 
            2. The latency and sizes of the best pyramidal MT cache 
            3. The latency and sizes of the best non-pyramidal MT cache if it exists 
        """
        max_t1_size = math.floor(cost/(mt_config[0]["price"]*cache_unit_size))
        for t1_size in range(1, max_t1_size+1):
            t1_cost = t1_size * cache_unit_size * mt_config[0]["price"]
            t2_cost = cost - t1_cost 
            t2_size = math.floor(t2_cost/(mt_config[1]["price"]*cache_unit_size))
            mean_lat = self.get_mt_mean_latency(t1_size*cache_unit_size, t2_size*cache_unit_size, lat_array)

            if t1_size < t2_size:
                if mean_lat < mt_p_lat:
                    mt_p_config = [t1_size, t2_size]
                    mt_p_lat = mean_lat 
            else:
                if mean_lat < mt_np_lat:
                    mt_np_config = [t1_size, t2_size]
                    mt_np_lat = mean_lat 
            
            # if the ST cache has already been evaluated then break;
            if t1_size == max_t1_size and t2_size == 0:
                st_lat = mean_lat
                break
        else:
            # since ST cahce has not been compute, evaluate the ST cache 
            mean_lat = self.get_mt_mean_latency(t1_size, t2_size, lat_array)
            st_lat = mean_lat
        
        return {
            "cost": cost,
            "st_t1": max_t1_size,
            "st_lat": st_lat,
            "mt_p_t1": mt_p_config[0],
            "mt_p_t2": mt_p_config[1],
            "mt_p_lat": mt_p_lat,
            "mt_np_t1": mt_np_config[0],
            "mt_np_t2": mt_np_config[1],
            "mt_np_lat": mt_np_lat
        }


    def plot_mrc(self, plot_path, max_cache_size=-1):
        """ Plot MRC of the RD histogram

        Params
        ------
        plot_path: output path (str)
        max_cache_size: max cache size (defaults to -1) (optional) (int)
        """
        pass 


    def plot_mt_mrc(self, t1_size, plot_path, max_cache_size=-1):
        """ Plot MT MRC of the RD histogram for a given T1 size 

        Params
        ------
        t1_size: size of Tier 1 (int)
        plot_path: output path (str)
        max_cache_size: max cache size (defaults to -1) (optional) (int)
        """
        pass 