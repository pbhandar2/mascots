import pathlib 
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