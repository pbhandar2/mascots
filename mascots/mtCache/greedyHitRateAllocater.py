import pathlib, time 
import numpy as np 
import pandas as pd 

from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist
from mascots.mtCache.mtCache import MTCache

class GreedyHitRateAllocater:

    def __init__(self, workload_name_list, device_config, size_array, data, log_path):
        self.data_dir = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k") 
        self.workload_name_list = workload_name_list 
        self.num_workloads = len(workload_name_list)
        self.size_array = size_array
        self.log_path_handle = log_path.open("w+")
        
        self.update_devices(device_config)

        """ dictionary contains DataFrame of each workload's exclusive cache performance for different cost """
        self.data = data
        self.workload_data = pd.read_csv("/research2/mtc/cp_traces/general/block_read_write_stats.csv")


    def update_devices(self, device_config, log_path=None):
        self.device_config = device_config 
        self.mt_label = "_".join([_["label"] for _ in self.device_config])
        self.metric_entry_list = [None]*self.num_workloads
        self.allocation_array = np.zeros(shape=(self.num_workloads, 2), dtype=int)
        self.metric_array = np.zeros(self.num_workloads, dtype=float)

        if log_path is not None:
            self.log_path_handle.close()
            self.log_path_handle = log_path.open("w+")


    def run(self, metric_name):
        print("Allocating for {}".format(self.mt_label))
        self.metric_name = metric_name
        for workload_index in range(len(self.workload_name_list)):
            self.load_metric(workload_index)

        while workload_index >= 0:
            workload_index = self.allocate()


    def get_remaining_size(self):
        total_allocation = np.sum(self.allocation_array, axis=0)
        return [self.size_array[0]-total_allocation[0], self.size_array[1]-total_allocation[1]]


    def allocate(self):
        """ If there is no reamining cache resources or no positive metric, return -1."""
        workload_index = np.argmax(self.metric_array)
        remaining_sizes = self.get_remaining_size()
        if (remaining_sizes[0] == 0 and remaining_sizes[1] == 0) or self.metric_array[workload_index] == -1:
            return -1 

        allocation_entry = self.metric_entry_list[workload_index]
        cur_allocation = self.allocation_array[workload_index]
        additional_allocation = allocation_entry["rd"].item() + 1 - np.sum(cur_allocation)

        assert(np.sum(remaining_sizes)>=additional_allocation)

        if remaining_sizes[0]>0:
            if remaining_sizes[0]>=additional_allocation:
                self.allocation_array[workload_index][0] += additional_allocation
                self.update_allocation_log(workload_index, 1, self.metric_entry_list[workload_index], additional_allocation)
                additional_allocation = 0
            else:
                self.allocation_array[workload_index][0] += remaining_sizes[0]
                additional_allocation -= remaining_sizes[0]
                self.update_allocation_log(workload_index, 1, self.metric_entry_list[workload_index], remaining_sizes[0])

        if remaining_sizes[1]>0 and additional_allocation>0:
            if remaining_sizes[1]>=additional_allocation:
                self.allocation_array[workload_index][1] += additional_allocation
                self.update_allocation_log(workload_index, 2, self.metric_entry_list[workload_index], additional_allocation)
                additional_allocation = 0
            else:
                self.allocation_array[workload_index][1] += remaining_sizes[1]
                additional_allocation -= remaining_sizes[1]
                self.update_allocation_log(workload_index, 2, self.metric_entry_list[workload_index], remaining_sizes[1])

        # recompute the metrics where needed 
        for workload_index in range(len(self.workload_name_list)):
            self.load_metric(workload_index)

        return workload_index 


    def update_allocation_log(self, workload_index, tier_num, allocation_entry, allocated_size):
        workload_name = self.workload_name_list[workload_index]
        cur_cost = self.device_config[0]["price"]*2560*self.allocation_array[workload_index][0] + \
            self.device_config[1]["price"]*2560*self.allocation_array[workload_index][1]

        log_entry = ",".join([workload_name, str(tier_num), str(self.allocation_array[workload_index][0]), \
            str(self.allocation_array[workload_index][1]), \
            str(cur_cost), str(allocation_entry[self.metric_name].item()), str(allocated_size)])

        print(log_entry)
        self.log_path_handle.write("{}\n".format(log_entry))

        
    def load_metric(self, workload_index):
        workload_name = self.workload_name_list[workload_index]
        total_remaining_size = np.sum(self.get_remaining_size())

        # details about current resource allocation to this workload 
        cur_allocation = self.allocation_array[workload_index]
        cur_allocation_entry = self.metric_entry_list[workload_index]

        # get workload details and process it based on the current resource allocated to the workload 
        df = self.data[self.workload_name_list[workload_index]]["df"].copy() 
        df = df[df["rd"]<(total_remaining_size+np.sum(cur_allocation))]
        size_array = df["rd"] - np.sum(self.allocation_array[workload_index])
        size_array = size_array + 1
        filter_df = df[size_array>0]

        if len(filter_df)>0:
            # This means there are resources that can be allocated 
            if np.sum(self.allocation_array[workload_index])>0:
                if self.metric_name == "hits_per_size":
                    total_hits = df.iloc[np.sum(self.allocation_array[workload_index])-1]["cum_total_hits"].item()
                    adjusted_total_hits = filter_df.loc[:, "cum_total_hits"] - total_hits # subtract from the dataframe
                    adjusted_hits_per_size = adjusted_total_hits/size_array # compute how much hits per additional cache size we will get 
                elif self.metric_name == "read_hits_per_size":
                    total_hits = df.iloc[np.sum(self.allocation_array[workload_index])-1]["cum_read_hits"].item()
                    adjusted_total_hits = filter_df.loc[:, "cum_read_hits"] - total_hits # subtract from the dataframe
                    adjusted_hits_per_size = adjusted_total_hits/size_array # compute how much hits per additional cache size we will get 
                cur_entry = filter_df.loc[adjusted_hits_per_size[(adjusted_hits_per_size==adjusted_hits_per_size.max()) & \
                    (adjusted_hits_per_size>0)].index, :]
            else:
                cur_entry = filter_df[filter_df[self.metric_name]==filter_df[self.metric_name].max()].iloc[0]
            
            print(cur_entry)
            assert(len(cur_entry) > 0)
            self.metric_array[workload_index] = cur_entry[self.metric_name].item()
            self.metric_entry_list[workload_index] = cur_entry
        else:
            # No resources left to be allocated 
            self.metric_array[workload_index] = -1 
            self.metric_entry_list[workload_index] = None


    def get_latency(self, summary_file):
        summary_handle = summary_file.open("w+")
        mt_cache = MTCache()
        total_lat = 0.0
        total_cost = 0.0
        for workload_index in range(len(self.workload_name_list)):
            workload_name = self.workload_name_list[workload_index]
            tier_1_size = self.allocation_array[workload_index][0]
            tier_2_size = self.allocation_array[workload_index][1]

            cost = self.device_config[0]["price"]*2560*tier_1_size + \
                self.device_config[1]["price"]*2560*tier_2_size

            hits, tier_lat, mean_lat = mt_cache.eval_exclusive_mt_cache(self.data[workload_name]["hist"], \
                tier_1_size, tier_2_size, self.device_config)

            total_lat += mean_lat 
            total_cost += cost 

            workload_string = "{},{},{},{},{}".format(workload_name, tier_1_size, tier_2_size, cost, mean_lat)
            summary_handle.write("{}\n".format(workload_string))

        remaining_sizes = self.get_remaining_size()
        overall_string = "Overall,{},{},{},{},{}".format(remaining_sizes[0], remaining_sizes[1], 
            total_cost, total_lat/len(self.workload_name_list), total_lat/total_cost)
        summary_handle.write("{}\n".format(overall_string))
        summary_handle.close()

        
        



