import numpy as np 
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

class MTCache:

    def get_t1_hits(self, rd_hist, t1_size):
        return np.sum(rd_hist[1:t1_size+1], axis=0)


    def get_t2_exclusive_hits(self, rd_hist, t1_size):
        t2_hits = np.zeros(shape=(len(rd_hist[t1_size+1:])+1, 2))
        t2_hits[1:, :] = np.cumsum(rd_hist[t1_size+1:], axis=0)
        return t2_hits 


    def get_exclusive_tier_lat_wb(self, mt_config, device_config):
        """ Get the latency of each tier in a multi-tier configuration.
        """

        assert(len(mt_config)==len(device_config)-1) # device config has extra slot for storage 

        mt_lat = np.zeros(shape=(len(mt_config)+1, 2)) # for a 2 tier system 
        cache_tiers = []
        for device_index, device_size in enumerate(mt_config):
            if device_size > 0:
                if len(cache_tiers) == 0:
                    # top tier so read and write latency is the read/write latency of tier 1 device 
                    read_lat = device_config[device_index]["read_lat"]
                    write_lat = device_config[device_index]["write_lat"]
                else:
                    # lower tiers 
                    cur_tier_lat = 0.0
                    # a read and write occurs at each tier above the one where there was the hit 
                    # device_index >= 1 because device_index = 0 is handled 
                    for _device_index in range(device_index):
                        cur_tier_lat += device_config[_device_index]["read_lat"] + device_config[_device_index]["write_lat"]
                    # read hit at tier "n" causes read and write from tier 1 to n 
                    read_lat = cur_tier_lat + device_config[device_index]["read_lat"] + device_config[device_index]["write_lat"]
                    # write hit at tier "n" causes read and write from tier 1 to n-1 and a write to tier n 
                    write_lat = cur_tier_lat + device_config[device_index]["write_lat"] 
                mt_lat[device_index] = np.array([read_lat, write_lat], dtype=float)
                cache_tiers.append(device_config[device_index])

        # Compute the latency of the backend storage
        if len(cache_tiers) == 0:
            miss_read_lat = device_config[-1]["read_lat"]
            miss_write_lat = device_config[-1]["write_lat"]
        elif len(cache_tiers) == 1:
            miss_read_lat = cache_tiers[0]["write_lat"] + device_config[-1]["read_lat"]
            miss_write_lat = cache_tiers[0]["write_lat"] 
        else:
            # read, write in all tiers except the last on a read miss 
            cur_tier_lat = 0.0
            for _device_index in range(len(cache_tiers)-1):
                cur_tier_lat += cache_tiers[_device_index]["read_lat"] + cache_tiers[_device_index]["write_lat"]

            miss_read_lat = cur_tier_lat + device_config[-1]["read_lat"] + cache_tiers[-1]["write_lat"]
            miss_write_lat = cur_tier_lat + cache_tiers[-1]["write_lat"]

        mt_lat[len(mt_config)] = np.array([miss_read_lat, miss_write_lat], dtype=float)
        return mt_lat


    def exhaustive_exclusive(self, rd_hist_file_path, bin_width, device_config_list, output_path):
        """ Exhaustive search of all possible exclusive cache configuration. 
        """

        print("Exhaustive exclusive: {},{},{}".format(rd_hist_file_path, bin_width, output_path))
        with output_path.open("a+") as f:

            # load the RD histogram 
            rd_hist = bin_rdhist(read_reuse_hist_file(rd_hist_file_path), bin_width)
            total_req = np.sum(rd_hist, axis=0)

            # iterate through all possible tier 1 and 2 sizes 
            max_rd = len(rd_hist) - 2 # one entry is for cold miss and starts from 0 
            for t1_size in range(max_rd):
                print("evaluating T1 size: {}".format(t1_size))
                t1_hits = self.get_t1_hits(rd_hist, t1_size)
                t2_hits_array = self.get_t2_exclusive_hits(rd_hist, t1_size)

                for t2_size in range(len(t2_hits_array)):
                    t2_hits = t2_hits_array[t2_size]
                    total_miss = total_req-t1_hits-t2_hits 

                    for device_config in device_config_list: 
                        mt_lat = self.get_exclusive_tier_lat_wb([t1_size, t2_size], device_config)
                        cache_hits = np.array([t1_hits, t2_hits, total_miss])
                        mean_lat = np.sum(np.multiply(cache_hits, mt_lat))/np.sum(total_req)

                        total_hits = np.sum(t1_hits+t2_hits)
                        hit_rate = np.sum(total_hits)/np.sum(total_req)

                        dollar_cost_config = bin_width * device_config[0]["price"] * t1_size + bin_width * device_config[1]["price"] * t2_size
                        performance_per_dollar = mean_lat/dollar_cost_config 

                        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            "{}_{}_{}".format(device_config[0]["label"], device_config[1]["label"], device_config[2]["label"]),
                            t1_size,
                            t2_size, 
                            t1_hits[0],
                            t1_hits[1],
                            t2_hits[0],
                            t2_hits[1],
                            total_miss[0],
                            total_miss[1],
                            mean_lat,
                            hit_rate,
                            dollar_cost_config,
                            performance_per_dollar
                        ))
