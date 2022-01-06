import pathlib, math 
import numpy as np 
import matplotlib.pyplot as plt
import logging 
logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


class RDHistProfiler:
    """ RDHistProfiler class allows user to perform single-tier
    and multi-tier cache analysis from a file containing the 
    reuse distance histogram. """


    def __init__(self, rd_hist_file_path, page_size=4096):  
        """ 
        Parameters
        ----------
        rd_hist_file_path: str
            path to the RD histogram file 
        page_size : int 
            page size in bytes (optional) (Default: 4096)
        """

        self.page_size = page_size
        self.file = pathlib.Path(rd_hist_file_path)

        self.data = np.loadtxt(rd_hist_file_path, delimiter=",", dtype=int)
        self.cold_miss = self.data[0] # record cold misses in a separate variable 
        self.data[0] = np.array([0,0]) # replace cold misses in the data array 

        self.read_count = self.data[:, 0].sum() + self.cold_miss[0]
        self.write_count = self.data[:, 1].sum() + self.cold_miss[1]
        self.req_count = self.read_count + self.write_count 
        self.hrc = self.data.cumsum()/self.req_count
        self.mrc = 1-self.hrc 
        self.max_cache_size = len(self.data) - 1 


    def get_page_cost(self, mt_config, index):
        """ Compute the cost of a page at a given tier index. 

        Parameters
        ----------
        mt_config : list
            list of JSONs representing cache tiers 
        indes : int 
            the index of the cache tier 
        
        Return
        ------
        page_cost : float 
            the cost of a page in dollar 
        """

        return self.page_size*mt_config[index]["cost"]/(mt_config[index]["size"]*1024*1024*1024)


    def get_exclusive_cache_hits(self, t1_size, t2_size):
        """ Get the number of cache hits at each tier and the remaining cache misses.

        Parameters
        ----------
        t1_size : int
            size of Tier 1 
        t2_size : int 
            size of Tier 2 

        Return
        ------
        hit_array : np.array
            array with read and write hits at each tier 
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

        Parameters
        ----------
        t1_size : int
            size of Tier 1 
        t2_size : int 
            size of Tier 2 
        lat_array : np.array
            array of read and write latency of each tier 

        Returns
        -------
        mean_latency : float
            mean latency of MT cache and given device latency values 
        """

        cache_hits = self.get_exclusive_cache_hits(t1_size, t2_size)
        total_latency = np.sum(np.multiply(cache_hits, lat_array))
        return total_latency/self.req_count


    def two_tier_exclusive_wb(self, mt_config):
        """ For a given MT configuration, return the latency of a 
        2-tier write-back exclusive MT cache.

        Parameters
        ----------
        mt_config : list
            list of dict containing properties of cache device at 
            each tier 

        Returns
        -------
        lat_array : np.array
            array of read and write latency of each cache tier 
        """

        lat_array = np.zeros([len(mt_config), 2], dtype=float) 

        # Tier 1 Read Lat = read from T1 
        lat_array[0][0] = mt_config[0]["read_lat"]
        # Tier 1 Write Lat = write to T1 
        lat_array[0][1] = mt_config[0]["write_lat"] 

        # Tier 2 Read Lat = read from T2 + write to T1 + read from T1 + write to T2 
        lat_array[1][0] = mt_config[1]["read_lat"] + mt_config[0]["write_lat"] \
            + mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
        # Tier 2 Write Lat = write to T1 + read from T1 + write to T2 
        lat_array[1][1] = mt_config[0]["write_lat"] + mt_config[0]["read_lat"] \
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

        Parameters
        ----------
        mt_config : list
            list of dict containing properties of cache device at 
            each tier 

        Returns
        -------
        lat_array : np.array
            array of read and write latency of each cache tier 
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


    def cost_eval_exclusive(self, mt_config, write_policy, cost, cache_unit_size, min_t2_size=350):
        """ Evaluate various cache configuration of a given cost and MT cache configuration. 

        Parameters
        ----------
        mt_config : list
            list of dict containing properties of cache device at 
            each tier 
        write_policy : str
            write policy of the MT cache 
        cost : float
            the cost of the cache 
        cache_unit_size: int
            the unit size of cache allocation 
        min_t2_size : int 
            minimum size of tier 2 cache in MB 

        Returns
        -------
        mt_cache_info: dict
            dict containing the following keys: 
                cost : float
                    cost of the cache in dollars 
                st_t1 : int 
                    max size of tier 1 scaled by cache unit size 
                st_lat : float
                    mean latency of an ST cache 
                mt_p_t1 : int
                    tier 1 size of pyramidal MT cache 
                mt_p_t2 : int
                    tier 2 size of pyramidal MT cahce 
                mt_p_lat : float
                    mean latency of pyramidal MT cache 
                mt_np_t1 : int
                    tier 1 size of non-pyramidal MT cache 
                mt_np_t2 : int
                    tier 2 size of non-pyramidal MT cache 
                mt_np_lat : float 
                    mean latency of non-pyramidal MT cache
        """

        if write_policy == "wb":
            lat_array = self.two_tier_exclusive_wb(mt_config)
        elif write_policy == "wt":
            lat_array = self.two_tier_exclusive_wb(mt_config)
        else:
            raise ValueError

        mt_p_config, mt_np_config = [0,0], [0,0]
        mt_p_lat, mt_np_lat, st_lat = math.inf, math.inf, math.inf

        t1_cost_per_byte = mt_config[0]["cost"]/(mt_config[0]["size"]*1e9)
        t1_cost_per_page = self.page_size * t1_cost_per_byte
        t1_cost_per_unit = t1_cost_per_page * cache_unit_size 

        t2_cost_per_byte = mt_config[1]["cost"]/(mt_config[1]["size"]*1e9)
        t2_cost_per_page = self.page_size * t2_cost_per_byte
        t2_cost_per_unit = t2_cost_per_page * cache_unit_size 

        """ For the given cost, find 
            1. The latency and size of an ST cache. 
            2. The latency and sizes of the best pyramidal MT cache 
            3. The latency and sizes of the best non-pyramidal MT cache if it exists 
        """
        max_t1_size = math.floor(cost/t1_cost_per_unit)
        for t1_size in range(1, max_t1_size+1):
            t1_cost = t1_size * t1_cost_per_unit
            t2_cost = cost - t1_cost 
            t2_size = math.floor(t2_cost/t2_cost_per_unit)

            if t2_size < min_t2_size:
                continue 

            mean_lat = self.get_mean_latency(t1_size*cache_unit_size, t2_size*cache_unit_size, lat_array)

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
            mean_lat = self.get_mean_latency(t1_size, t2_size, lat_array)
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


    def plot_mrc(self, plot_path, max_cache_size=-1, num_x_labels=10, hrc_flag=False):
        """ Plot MRC of the RD histogram

        Parameters
        ----------
        plot_path: str
            output path 
        max_cache_size: int
            max cache size (Default: -1) (optional) 
        num_x_labels: int
            number of xlabels (Default: 10) (optional) 
        """
        
        plot_path = pathlib.Path(plot_path)
        len_mrc = len(self.mrc) if max_cache_size == -1 else min(len(self.mrc), max_cache_size)
        fig, ax = plt.subplots(figsize=(14,7))

        if hrc_flag:
            ax.plot(self.hrc)
        else:
            ax.plot(self.mrc)

        x_ticks = np.linspace(0, len_mrc, num=num_x_labels, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(["{}".format(int(_*self.page_size/(1024*1024))) for _ in x_ticks])
        ax.set_xlabel("Cache Size (MB)")

        if hrc_flag:
            ax.set_ylabel("Hit Rate")
        else:
            ax.set_ylabel("Miss Rate")

        ax.set_title("Workload: {}, Read Cold Miss Rate: {:.3f} \n Ops: {:.1f}M Total IO: {}GB Write: {:.1f}%".format(
            plot_path.stem, 
            self.cold_miss[0]/self.read_count,
            self.req_count/1e6,
            math.ceil(self.req_count*self.page_size/(1024*1024*1024)),
            self.write_count/self.req_count))

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


    def plot_mhrc(self, plot_path, max_cache_size=-1):
        """ Plot MHRC from a RD histogram. 

        Parameters
        ----------
        plot_path: output path (str)
        max_cache_size: max cache size (Default: -1) (optional) (int)
        """

        plot_path = pathlib.Path(plot_path)
        len_mrc = len(self.mrc) if max_cache_size == -1 else min(len(self.mrc), max_cache_size)
        fig, ax = plt.subplots(figsize=(14,7))

        """ At each cache size, the ratio of ( misses + write_count )/hits 
        """

