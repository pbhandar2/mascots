import pathlib
import math  
import numpy as np 

class RDHistProfiler:
    def __init__(self, rd_hist_file_path):  
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
        """ Get the number of cache hits at each tier and the remaining cache misses 
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


    def get_mt_mean_latency(self, t1_size, t2_size, lat_array):
        """ Compute the mean latency of an MT cache with size and latency of each tier. 
        """

        cache_hits = self.get_exclusive_cache_hits(t1_size, t2_size)
        total_latency = np.sum(np.multiply(cache_hits, lat_array))
        return total_latency/self.req_count


    def cost_eval_exclusive(self, mt_config, write_policy, output_dir, cost, cache_unit_size):
        if write_policy == "wb":
            lat_array = self.two_tier_exclusive_wb(mt_config)
        elif write_policy == "wt":
            lat_array = self.two_tier_exclusive_wb(mt_config)
        else:
            raise ValueError

        max_t1_size = math.floor(cost/mt_config[0]["price"])
        for t1_size in range(1, max_t1_size+1):
            t1_cost = t1_size * cache_unit_size * mt_config[0]["price"]
            t2_cost = cost - t1_cost 
            t2_size = math.floor(t2_cost/(mt_config[1]["price"]*cache_unit_size))
            wb_mean_lat = self.get_mt_mean_latency(t1_size, t2_size, wb_lat_array)
            wt_mean_lat = self.get_mt_mean_latency(t1_size, t2_size, wt_lat_array)




            


























    def two_tier_exclusive_wb(self, mt_config):
        lat_array = np.zeros([2, len(mt_config)], dtype=float) 

        # tier 1 latency 
        lat_array[0][0] = mt_config[0]["read_lat"]
        lat_array[0][1] = mt_config[0]["write_lat"] 

        # tier 2 latency 
        lat_array[1][0] = mt_config[1]["read_lat"] + mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
        lat_array[1][1] = self.config[0]["write_lat"] + mt_config[0]["read_lat"] \
            + mt_config[1]["write_lat"] 

        # miss latency 
        lat_array[2][0] = mt_config[-1]["read_lat"] + mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
        lat_array[2][1] = mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]

        return lat_array


    def two_tier_exclusive_wt(self, mt_config):
        lat_array = self.two_tier_exclusive_wb(mt_config)
        for i in range(len(mt_config)):
            lat_array[i][1] += self.config[-1]["write_lat"]
        return lat_array

    
    def get_report(self, mt_list, output_dir):
        # for each mt cache setup a directory of output in output_dir

        for mt_cache in mt_list:
            output_dir = pathlib.Path(output_dir)
            # get the mt cache tag or name 
            # create the directory 
            pass



        # max size of T1
        max_t1_size = 100
        for i in range(1, max_t1_size+1):
            pass 

        pass 


    def plot_mrc(self, cache_size, output_file_path):
        """ Plot MRC from the reuse distance histogram 
        """
        pass 


    def get_size_for_miss_rate(self, miss_rate):
        """ The size of cache required to meet the miss rate provided. If miss rate can't be met, 
            the best possible case is returned. 
        """
        pass


    def get_miss_rate(self, cache_size):
        """ Get the miss rate for a given cache size. 
        """
        pass