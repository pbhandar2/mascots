import json, math, itertools
import numpy as np 

class MTConfigLib:
    """ MT Configuration class handles device specific computations such as 
        latency of write-back, write-through caches, or max size of cache 
        tier for a given cost. 

        Attributes
        ----------
        page_size : int 
            page size in bytes (Default: 4096)
        metadata_size : int 
            size of metadata per page (Default: 31)
    """

    def __init__(self, page_size=4096, metadata_size=31):
        """
        Paramters
        ---------
        page_size : int 
            page size in bytes (optional) (Default: 4096)
        """

        self.page_size = page_size
        self.metadata_size = metadata_size


    def get_mt_label(self, mt_config):
        """ Generate a unique label for an MT cache 

        Parameters
        ----------
        mt_config : dict 
            dict containing properties of devices at each tier 

        Return
        ------
        mt_label : str 
            unique label of MT cache based on device labels at each tier 
        """
        
        return "_".join([device["label"] for device in mt_config])


    def get_max_cost(self, mt_config, max_cache_size, allocation_unit):
        """ Get the maximum cost of a workload, which is defined 
            as the cost large enough to fit the entire working 
            set in the fastest, expensive cache device 

            Parameters
            ----------
            mt_config : list 
                list of dictionaries with device specification of each tier 
            max_cache_size : int 
                maximum size of cache in allocation units 
            allocation_unit : int 
                the number of pages in an allocation unit 
        """
        
        return math.ceil(max_cache_size * self.get_unit_cost(mt_config, 0, allocation_unit))
    

    def get_page_cost(self, mt_config, tier_index, metadata_flag=True):
        """ Compute the cost of a page at a given tier index. 

        Parameters
        ----------
        mt_config : list
            list of JSONs representing cache tiers 
        indes : int 
            the index of the cache tier 
        metadata_flag : bool    
            indicates where the cost of metadata is taken into account (Default: True)
        
        Return
        ------
        page_cost : float 
            the cost of a page in dollar 
        """

        metadata_cost = 0 
        data_cost = self.page_size * (mt_config[tier_index]["cost"]/(mt_config[tier_index]["size"]*1e9))
        if metadata_flag:
            metadata_cost = self.metadata_size * mt_config[0]["cost"]/(mt_config[0]["size"]*1e9)
        return data_cost + metadata_cost


    def get_unit_cost(self, mt_config, tier_index, cache_unit_size):
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

        return self.get_page_cost(mt_config, tier_index) * cache_unit_size

    
    def get_cache_cost(self, mt_config, size_array, cache_unit_size):
        """ Get the cost of the cache based on device cost and tier sizes 

        Parameters
        ----------
        mt_config : dict 
            dict with device specification of each tier 
        size_array : np.array 
            array containing the size of each tier 

        Return
        ------
        cache_cost : float 
            the total cost of the cache 
        """

        num_tiers = len(size_array)
        cost_array = np.array([self.get_unit_cost(mt_config, tier_index, cache_unit_size) 
            for tier_index in range(num_tiers)])
        return np.sum(np.prod((size_array, cost_array), axis=0))


    def get_max_t2_size(self, mt_config, t1_size, cost, allocation_unit):
        """ Get the max size of tier 2 cache given tier 1 size 
            and cost. 

            Parameters
            ----------
            mt_config : list
                list with device specifications (dict) of each tier 
            t1_size : int 
                size of tier 1 
            cost : int 
                cost in dollars 
            allocation_unit : int 
                the number of pages per allocation 

            Return
            ------
            t2_size : int 
                the max size of tier 2 possible in the given scenario 
        """
        t1_cost = t1_size * self.get_unit_cost(mt_config, 0, 
                    allocation_unit) 
        assert(t1_cost<=cost)
        t2_cost = cost - t1_cost
        return self.get_max_tier_size(mt_config, 1, allocation_unit, t2_cost)


    def get_max_tier_size(self, mt_config, tier_index, cache_unit_size, cost):
        """ Get the maximum size of a cache tier for a given cost 

        Parameters
        ----------
        mt_config : list 
            list of dict containing device specification of each tier 
        tier_index : int 
            the index of cache tier (starts from 0)
        cache_unit_size : int 
            the number of pages in the unit of cache allocation 
        cost : float 
            the cost of the cache tier 

        Return 
        ------
        max_tier_size : int 
            the maximum size of cache tier based on cost 
        """
        
        return math.floor(cost/self.get_unit_cost(mt_config, tier_index, cache_unit_size))


    def get_overhead_gain_ratio(self, mt_config, adm_policy="ex"):
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

        if adm_policy == "ex":
            overhead = mt_config[0]["read_lat"] + mt_config[1]["write_lat"]
            gain = mt_config[2]["read_lat"] - mt_config[1]["read_lat"] - mt_config[0]["read_lat"] \
                - mt_config[1]["write_lat"]
            og_ratio = overhead/gain 
        elif adm_policy == "in":
            overhead = mt_config[1]["write_lat"]
            gain = mt_config[2]["read_lat"] - mt_config[1]["read_lat"] - mt_config[1]["write_lat"]
            og_ratio = overhead/gain 
        else:
            raise ValueError("{} not an accepted admission policy. exclusive or inclusive".format(adm_policy))
        return og_ratio


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

        num_cache_tiers = len(mt_config) - 1
        lat_array = np.zeros([len(mt_config), 2], dtype=float) 

        lat_array[0][0] = mt_config[0]["read_lat"]
        lat_array[0][1] = mt_config[0]["write_lat"] 

        """ Computing the latency of cache tier N where N > 0,
            read hit -> when there is a read hit at tier N, 
                1. data is read from tier N and stored in tier 0 which causes an eviction 
                2. data being evicted is read from tier 0 and written to tier 1 
                3. Step 2 happens until tier N-1, there is no eviction from tier N 
                    since we assume data is already persisted
                4. A page has been read and written at each tier [0..N]
            write hit -> where there is a write hit at tier N, 
                1. data is written to tier 0 which causes an eviction 
                2. data being evicted is read from tier 0 and written to tier 1 
                3. Step 2 happens until tier N-1, there is no eviction from tier N 
                    since we assume data is already persisted 
                4. A page has been read and written at tier [0..N-1] and only written in tier N 
        """
        for tier_index in range(1, num_cache_tiers):
            for _temp_tier in range(tier_index):
                lat_array[tier_index][0] += mt_config[_temp_tier]["read_lat"] + mt_config[_temp_tier]["write_lat"]
                lat_array[tier_index][1] += mt_config[_temp_tier]["read_lat"] + mt_config[_temp_tier]["write_lat"]
            else:
                lat_array[tier_index][0] += mt_config[tier_index]["read_lat"] + mt_config[tier_index]["write_lat"]
                lat_array[tier_index][1] += mt_config[tier_index]["write_lat"]

        """ Computing storage latency of read and write miss. 
            read miss -> when there is a read miss, 
                1. data is read from storage and stored in tier 0 which causes an eviction 
                2. data being evicted is read from tier 0 and written to tier 1 
                3. Step 2 happens until tier N-1, there is no eviction from tier N 
                    since we assume data is already persisted
                4. A page has been read and written at each tier [0..N] and read from storage 
                5. Therefore, total latency is the sum of latency of cache tier N and read from storage 
            write miss -> when there is a write miss, 
                1. data is written to tier 0 which causes an eviction 
                2. data being evicted is read from tier 0 and written to tier 1 
                3. Step 2 happens until tier N-1, there is no eviction from tier N 
                    since we assume data is already persisted 
                4. A page has been read and written at tier [0..N-1] and only written in tier N 
                5. Therefore, total latency is the same as a write hit at tier N 
                6. No write to storage because we are using a write-back cache and assume data is persisted async
        """
        lat_array[num_cache_tiers][0] = mt_config[-1]["read_lat"] + lat_array[num_cache_tiers-1][1]
        lat_array[num_cache_tiers][1] = lat_array[num_cache_tiers-1][1]

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
        """ Latency of write-through cache is the same as the write-back cache 
            except the latency of any write requests (hit or miss) has an 
            additional latency of writing to storage. 
        """
        for i in range(len(mt_config)):
            lat_array[i][1] += mt_config[-1]["write_lat"]
        return lat_array


    def generate_exclusive_mt_cache(self, device_list_file, 
        dram_only_t1=True, 
        no_hdd_cache=True, 
        max_tiers=5, 
        min_percent_diff_storage_cache=40):
        """ Generate all possible exclusive caches from a file with a list of devices

        Parameters
        ----------
        device_list_file : str 
            path to file containing the list of device and its properties 
        dram_only_t1 : bool
            flag to force tier 1 cache as DRAM (optional) (Default: True)
        no_hdd_cache : bool
            flag to force HDD to only be used as storage (optional) (Default: True)
        max_tiers : int 
            max cache tiers in MT cache generated(optional) (Default: 5)

        Return 
        ------
        mt_dict : dict
            dict with MT caches indexed by number of tiers
        """

        device_list = None
        with open(device_list_file) as f:
            device_list = json.load(f)
        
        mt_dict = {}
        num_tiers = 2
        while True:
            num_valid_config = 0
            mt_list = []
            device_combinations = itertools.combinations(device_list, num_tiers+1)
            for mt_cache in device_combinations:
                if dram_only_t1:
                    if mt_cache[0]["type"] != "DRAM":
                        continue 
                
                """ The requirement for an MT cache is that the sum of read and 
                    write latency of a cache tier N is always less than the sum 
                    of read and write latency of a cache of a cache tier < N. 
                """
                prev_latency_sum = mt_cache[0]["read_lat"]+mt_cache[0]["write_lat"]
                for tier_index in range(1, num_tiers+1):
                    latency_sum = mt_cache[tier_index]["read_lat"]+mt_cache[tier_index]["write_lat"]
                    if no_hdd_cache:
                        if tier_index < num_tiers:
                            if mt_cache[tier_index]["type"] == "HDD":
                                break
                        elif tier_index == num_tiers:
                            # check if storage is sufficiently slower than last tier of cache 
                            if prev_latency_sum >= latency_sum*min_percent_diff_storage_cache/100:
                                break 

                    if latency_sum < prev_latency_sum:
                        break
                    prev_latency_sum = latency_sum
                else:
                    # this means all requirements were fulfilled 
                    mt_list.append(mt_cache)
                    num_valid_config += 1

            if num_valid_config > 0:
                mt_dict[num_tiers] = mt_list 
                num_tiers += 1
            else:
                break

            if num_tiers > max_tiers:
                break 
        
        return mt_dict
