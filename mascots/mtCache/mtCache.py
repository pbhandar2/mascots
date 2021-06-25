import time, json, math
import numpy as np 
import pandas as pd 
from collections import defaultdict 
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

class MTCache:


    def cache_name_from_config(self, device_config):
        return "_".join([_["label"] for _ in device_config])

    
    def get_cost_entry(self, t1, t2, mean_lat, base_lat, dollar_cost_config):
        data = {
            "t1": t1,
            "t2": t2,
            "min_lat": mean_lat,
            "lat_reduce": base_lat - mean_lat,
            "lat_reduce_per_dollar": (base_lat - mean_lat)/dollar_cost_config
        }
        return data 


    def get_st_entry(self, st_size, st_lat, base_lat, dollar_cost_config):
        data = {
            "st_size": st_size,
            "min_lat": st_lat,
            "lat_reduce": base_lat - st_lat,
            "lat_reduce_per_dollar": (base_lat - st_lat)/dollar_cost_config,
        }
        return data 


    def get_t1_hits(self, rd_hist, t1_size):
        return np.sum(rd_hist[1:t1_size+1], axis=0)


    def get_t2_exclusive_hits(self, rd_hist, t1_size):
        t2_hits = np.zeros(shape=(len(rd_hist[t1_size+1:])+1, 2))
        t2_hits[1:, :] = np.cumsum(rd_hist[t1_size+1:], axis=0)
        return t2_hits 


    def get_no_cache_latency(self, device_config):
        # Latency where there is no cache 
        mt_lat = np.zeros(shape=(len(device_config), 2)) 
        mt_lat[-1][0] = device_config[-1]["read_lat"]
        mt_lat[-1][1] = device_config[-1]["write_lat"]
        return mt_lat 


    def get_single_tier_lat_wb(self, mt_config, device_config):
        device_index = np.nonzero(mt_config)[0][0]
        mt_lat = np.zeros(shape=(len(mt_config)+1, 2)) 
        mt_lat[device_index][0] = device_config[device_index]["read_lat"]
        mt_lat[device_index][1] = device_config[device_index]["write_lat"]

        mt_lat[-1][0] = device_config[device_index]["write_lat"] + device_config[-1]["read_lat"]
        mt_lat[-1][1] = device_config[device_index]["write_lat"]
        return mt_lat


    def get_single_tier_lat_wt(self, mt_config, device_config):
        device_index = np.nonzero(mt_config)[0][0]
        mt_lat = self.get_single_tier_lat_wb(mt_config, device_config)
        mt_lat[device_index][1] += device_config[-1]["write_lat"]
        mt_lat[-1][1] += device_config[-1]["write_lat"]
        return mt_lat 
            

    def get_exclusive_tier_lat_wb(self, mt_config, device_config):
        """ Get the latency of each tier in an exclusive multi-tier configuration.
        """

        num_cache_devices = len(mt_config)
        assert(num_cache_devices==len(device_config)-1) # device config has extra slot for storage 

        # check if there is a cache size=0, representing a single-tier configuration 
        count_nonzero_cache_size = np.count_nonzero(mt_config)

        mt_lat = np.zeros(shape=(len(mt_config)+1, 2)) 
        if count_nonzero_cache_size == 0:
            # this means no cache latency of serving everything from hard disk 
            mt_lat = self.get_no_cache_latency(device_config)
        elif count_nonzero_cache_size == 1:
            # this means latency of a multi-tier cache 
            mt_lat = self.get_single_tier_lat_wb(mt_config, device_config)
        else:
            # this is a multi-tier cache 
            num_tiers = 0 
            cache_tiers = []
            for device_index, device_size in enumerate(mt_config):
                if device_size > 0:
                    if num_tiers == 0:
                        # top tier so read and write latency is the read/write latency of tier 1 device 
                        read_lat = device_config[device_index]["read_lat"]
                        write_lat = device_config[device_index]["write_lat"]
                    else:
                        # lower tiers 
                        lat_sum = 0.0 
                        # read and write in all tiers above the current tier 
                        for _device_index in range(num_tiers):
                            lat_sum += device_config[_device_index]["read_lat"] + device_config[_device_index]["write_lat"]
                        read_lat = lat_sum + device_config[device_index]["read_lat"] + device_config[device_index]["write_lat"]
                        write_lat = lat_sum + device_config[device_index]["write_lat"]

                    mt_lat[device_index] = np.array([read_lat, write_lat], dtype=float)
                    num_tiers += 1
                    cache_tiers.append(device_config[device_index])

            # read, write in all tiers except the last on a read miss 
            lat_sum = 0.0
            for _device_index in range(len(cache_tiers)-1):
                lat_sum += device_config[_device_index]["read_lat"] + device_config[_device_index]["write_lat"]

            mt_lat[-1][0] = lat_sum + device_config[-1]["read_lat"] + cache_tiers[-1]["write_lat"]
            mt_lat[-1][1] = lat_sum + cache_tiers[-1]["write_lat"]

        return mt_lat 


    def get_exclusive_tier_lat_wt(self, mt_config, device_config):
        """ Get the latency of each tier in a MT configuration with write-through policy. 
        """
        count_nonzero_cache_size = np.count_nonzero(mt_config)
        if count_nonzero_cache_size == 0:
            # this means no cache latency of serving everything from hard disk 
            mt_lat = self.get_no_cache_latency(device_config)
        elif count_nonzero_cache_size == 1:
            # this means latency of a multi-tier cache 
            mt_lat = self.get_single_tier_lat_wt(mt_config, device_config)
        else:
            mt_lat = self.get_exclusive_tier_lat_wb(mt_config, device_config)
            for nonzero_index in np.nonzero(mt_config)[0]:
                mt_lat[nonzero_index][1] += device_config[-1]["write_lat"]
            mt_lat[-1][1] += device_config[-1]["write_lat"]
        return mt_lat 


    def get_mean_lat(self, mt_lat, cache_hits, total_req):
        return np.sum(np.multiply(cache_hits, mt_lat))/np.sum(total_req)


    def analyze_t2_exclusive(self, rd_hist, t1_size, bin_width, device_config_list):
        MAX_T2_SIZE_GB = 200
        num_4k_page_in_max_t2_size = MAX_T2_SIZE_GB*1024*256
        max_t2_size = math.ceil(num_4k_page_in_max_t2_size/bin_width)

        total_req = np.sum(rd_hist, axis=0)
        t1_hits = self.get_t1_hits(rd_hist, t1_size)
        t2_hits_array = self.get_t2_exclusive_hits(rd_hist, t1_size)

        mean_lat_base_array = np.zeros(len(device_config_list), dtype=float)
        for device_index, device_config in enumerate(device_config_list):
            # Base Statistics
            base_mt_lat = self.get_exclusive_tier_lat_wb([0, 0], device_config)
            base_cache_hits = np.array([[0,0],[0,0], np.sum(rd_hist, axis=0)])
            mean_lat_base = np.sum(np.multiply(base_cache_hits, base_mt_lat))/np.sum(total_req)
            mean_lat_base_array[device_index] = mean_lat_base

        results = defaultdict(list)
        for t2_size in range(min(len(t2_hits_array), max_t2_size)):
            t2_hits = t2_hits_array[t2_size]
            total_miss = total_req - t1_hits - t2_hits
            cache_hits = np.array([t1_hits, t2_hits, total_miss])

            for device_index, device_config in enumerate(device_config_list): 
                cache_name = self.cache_name_from_config(device_config)
                mean_lat_base = mean_lat_base_array[device_index]

                # Write back 
                mt_lat_wb = self.get_exclusive_tier_lat_wb([t1_size, t2_size], device_config)
                mean_lat_wb = np.sum(np.multiply(cache_hits, mt_lat_wb))/np.sum(total_req)

                # Write Through 
                mt_lat_wt = self.get_exclusive_tier_lat_wt([t1_size, t2_size], device_config)
                mean_lat_wt = np.sum(np.multiply(cache_hits, mt_lat_wt))/np.sum(total_req)

                cost = bin_width * device_config[0]["price"] * t1_size + \
                    bin_width * device_config[1]["price"] * t2_size
                total_cache_hits = np.array(t1_hits+t2_hits).sum()
                total_req = np.sum(total_req)

                results[cache_name].append(np.array([t1_size, t2_size, cost, mean_lat_wb, mean_lat_wt, \
                    mean_lat_base-mean_lat_wb, mean_lat_base-mean_lat_wt, total_cache_hits/total_req], dtype=float))

        header_array = ["t1", "t2", "cost", "mean_lat_wb", "mean_lat_wt", "d_lat_wb", "d_lat_wt", "hit_rate"]
        df_dict = {}
        for cache_name in results:
            df_dict[cache_name] = pd.DataFrame(results[cache_name], columns=header_array)
        return df_dict

        





    def exhaustive_exclusive(self, rd_hist_file_path, bin_width, device_config_list, output_dir, cost_output_dir):
        """ Exhaustive search of all possible exclusive cache configuration. 
        """
        timer = time.time()
        print("Exhaustive exclusive: {},{},{}".format(rd_hist_file_path, bin_width, output_dir))
        
        # open a new file to write for each device configuration to evaluate 
        fd_array = []
        cost_file_fd_array = []
        st_cost_file_fd_array = []
        cache_name_array = []
        cost_data = {}
        st_data = {}
        max_dollar_cost_config = {}
        for device_index, device_config in enumerate(device_config_list):
            cache_name = self.cache_name_from_config(device_config)
            output_path = output_dir.joinpath("{}.csv".format(cache_name))
            cost_output_path = cost_output_dir.joinpath("{}.csv".format(cache_name))
            st_cost_output_path = cost_output_dir.joinpath("st_{}.csv".format(cache_name))
            cache_name_array.append(cache_name)
            cost_data[cache_name] = {}
            st_data[cache_name] = {}
            max_dollar_cost_config[cache_name] = 0.0
            fd_array.append(output_path.open("w+"))
            cost_file_fd_array.append(cost_output_path.open("w+"))
            st_cost_file_fd_array.append(st_cost_output_path.open("w+"))

        # load the RD histogram 
        rd_hist = bin_rdhist(read_reuse_hist_file(rd_hist_file_path), bin_width)
        total_req = np.sum(rd_hist, axis=0)

        # iterate through all possible tier 1 and 2 sizes 
        max_rd = len(rd_hist) - 2 # one entry is for cold miss and starts from 0 
        base_lat_wb = None 
        base_lat_wt = None 
        
        for t1_size in range(max_rd+1):

            cur_time = time.time()
            time_elasped = cur_time - timer
            if time_elasped > 300:
                if t1_size == 0:
                    remaining = 100
                else:
                    remaining = time_elasped/(t1_size/max_rd)
                # workload_name, percentage completion, time elasped, est time remaining 
                print("W: {}, Done: {}, Time: {}, Est Remaining: {}".format(
                    rd_hist_file_path.stem,
                    t1_size/max_rd,
                    time_elasped,
                    remaining 
                ))
                timer = cur_time 

            t1_hits = self.get_t1_hits(rd_hist, t1_size)
            t2_hits_array = self.get_t2_exclusive_hits(rd_hist, t1_size)

            for t2_size in range(len(t2_hits_array)):
                t2_hits = t2_hits_array[t2_size]
                total_miss = total_req-t1_hits-t2_hits 

                for device_index, device_config in enumerate(device_config_list): 
                    cache_hits = np.array([t1_hits, t2_hits, total_miss])

                    mt_lat_wb = self.get_exclusive_tier_lat_wb([t1_size, t2_size], device_config)
                    mean_lat_wb = np.sum(np.multiply(cache_hits, mt_lat_wb))/np.sum(total_req)

                    mt_lat_wt = self.get_exclusive_tier_lat_wt([t1_size, t2_size], device_config)
                    mean_lat_wt = np.sum(np.multiply(cache_hits, mt_lat_wt))/np.sum(total_req)

                    # base latency value 
                    if t1_size == 0 and t2_size == 0:
                        base_lat_wb = mean_lat_wb 
                        base_lat_wt = mean_lat_wt
                        assert(mean_lat_wb == mean_lat_wt)

                    total_hits = np.sum(t1_hits+t2_hits)
                    hit_rate = np.sum(total_hits)/np.sum(total_req)

                    dollar_cost_config = bin_width * device_config[0]["price"] * t1_size + \
                        bin_width * device_config[1]["price"] * t2_size

                    if dollar_cost_config > max_dollar_cost_config[cache_name_array[device_index]]:
                        max_dollar_cost_config[cache_name_array[device_index]] = dollar_cost_config 

                    # write the details to file 
                    fd_array[device_index].write("{},{},{},{},{},{}\n".format(
                        t1_size,
                        t2_size, 
                        mean_lat_wb,
                        mean_lat_wt,
                        hit_rate,
                        dollar_cost_config
                    ))

                    # update and store the cost data 
                    dollar_cost_config_entry = math.ceil(dollar_cost_config)
                    cur_cache_name = cache_name_array[device_index]
                    if dollar_cost_config_entry in cost_data[cur_cache_name]:
                        if cost_data[cur_cache_name][dollar_cost_config_entry]["wb"]["min_lat"] > mean_lat_wb:
                            wb_cost_entry = self.get_cost_entry(t1_size, t2_size, mean_lat_wb, base_lat_wb, dollar_cost_config)
                            cost_data[cur_cache_name][dollar_cost_config_entry]["wb"] = wb_cost_entry

                        if cost_data[cur_cache_name][dollar_cost_config_entry]["wt"]["min_lat"] > mean_lat_wt:
                            wt_cost_entry = self.get_cost_entry(t1_size, t2_size, mean_lat_wt, base_lat_wt, dollar_cost_config)
                            cost_data[cur_cache_name][dollar_cost_config_entry]["wt"] = wt_cost_entry
                    else:
                        wb_cost_entry = self.get_cost_entry(t1_size, t2_size, mean_lat_wb, base_lat_wb, dollar_cost_config)
                        wt_cost_entry = self.get_cost_entry(t1_size, t2_size, mean_lat_wt, base_lat_wt, dollar_cost_config)
                        cost_data_entry = {
                            "wb": wb_cost_entry,
                            "wt": wt_cost_entry
                        }
                        cost_data[cur_cache_name][dollar_cost_config_entry] = cost_data_entry
                    
                    # collect the best single tier data 
                    if t2_size == 0:
                        if dollar_cost_config_entry in st_data[cur_cache_name]:
                            if st_data[cur_cache_name][dollar_cost_config_entry]["min_lat"] > mean_lat_wb:
                                if st_data[cur_cache_name][dollar_cost_config_entry]["min_lat"] > mean_lat_wb:
                                    st_entry = self.get_st_entry(t1_size, mean_lat_wb, base_lat_wb, dollar_cost_config)
                        else:
                            st_data_entry = self.get_st_entry(t1_size, mean_lat_wb, base_lat_wb, dollar_cost_config)
                            st_data[cur_cache_name][dollar_cost_config_entry] = st_data_entry

        # now accumulate the cost and st data and write it out to a file 
        for device_index, device_config in enumerate(device_config_list):
            for cur_dollar_cost in range(1, math.ceil(max_dollar_cost_config[cache_name_array[device_index]])+1):
                cur_cache_name = cache_name_array[device_index]
                cur_cost_data = cost_data[cur_cache_name][cur_dollar_cost]

                cost_file_fd_array[device_index].write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    cur_dollar_cost,
                    cur_cost_data["wb"]["t1"],
                    cur_cost_data["wb"]["t2"],
                    cur_cost_data["wb"]["min_lat"],
                    cur_cost_data["wb"]["lat_reduce"],
                    cur_cost_data["wb"]["lat_reduce_per_dollar"],
                    cur_cost_data["wt"]["t1"],
                    cur_cost_data["wt"]["t2"],
                    cur_cost_data["wt"]["min_lat"],
                    cur_cost_data["wt"]["lat_reduce"],
                    cur_cost_data["wt"]["lat_reduce_per_dollar"]
                ))

                if cur_dollar_cost in st_data[cur_cache_name]:
                    cur_st_data = st_data[cur_cache_name][cur_dollar_cost]
                    st_cost_file_fd_array[device_index].write("{},{},{},{},{}\n".format(
                        cur_dollar_cost,
                        cur_st_data["st_size"],
                        cur_st_data["min_lat"],
                        cur_st_data["lat_reduce"],
                        cur_st_data["lat_reduce_per_dollar"]
                    ))

                

