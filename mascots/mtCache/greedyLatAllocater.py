import pathlib, time 
import numpy as np 
import pandas as pd 


from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist
from mascots.mtCache.mtCache import MTCache


class GreedyLatAllocater:

    def __init__(self, workload_name_list, device_config, mt_size_array, data, log_path, mt_type="wb"):
        self.data_dir = pathlib.Path("/research2/mtc/cp_traces/mascots/cost") 
        self.workload_name_list = workload_name_list 
        self.num_workloads = len(workload_name_list)
        self.size_array = mt_size_array
        self.log_path_handle = log_path.open("w+")
        self.mt_type = mt_type
        
        self.update_devices(device_config)

        """ dictionary contains DataFrame of each workload's exclusive cache performance for different cost """
        self.data = data
        self.workload_data = pd.read_csv("/research2/mtc/cp_traces/general/block_read_write_stats.csv")


    def update_devices(self, device_config, log_path=None):
        self.device_config = device_config 
        self.mt_label = "_".join([_["label"] for _ in self.device_config])
        self.metric_entry_list = [None]*self.num_workloads
        self.allocation_entry_list = [None]*self.num_workloads
        self.allocation_array = np.zeros(shape=(self.num_workloads, 2), dtype=int)
        self.metric_array = np.zeros(self.num_workloads, dtype=float)
        self.cost_array = np.zeros(self.num_workloads, dtype=int)

        if log_path is not None:
            self.log_path_handle.close()
            self.log_path_handle = log_path.open("w+")


    def get_remaining_size(self):
        total_allocation = np.sum(self.allocation_array, axis=0)
        return [self.size_array[0]-total_allocation[0], self.size_array[1]-total_allocation[1]]


    def run(self):
        print("GREEDY LAT BASED ALLOCATION: {}".format(self.mt_label))
        # load the data and metric of each workload 
        for index in range(len(self.workload_name_list)):
            self.load_metric(index)

        allocation_entry = self.allocate()
        while allocation_entry is not None:
            for index in range(len(self.workload_name_list)):
                self.load_metric(index)
            allocation_entry = self.allocate()

    def load_metric(self, workload_index):
        """ Load the value of the metric to be maximized for a given workload 
        """

        # The amount of additional T1, T2 and cost for each cache allocation
        main_df = self.data[self.workload_name_list[workload_index]]["df"].copy()
        d_t1 = main_df["{}_t1".format(self.mt_type)] - self.allocation_array[workload_index][0]
        d_t2 = main_df["{}_t2".format(self.mt_type)] - self.allocation_array[workload_index][1]
        d_c = main_df["c"] - self.cost_array[workload_index]

        # The amount of cache remaining 
        total_allocation = np.sum(self.allocation_array, axis=0)
        t1_remaining = self.size_array[0] - total_allocation[0]
        t2_remaining = self.size_array[1] - total_allocation[1]

        assert(t1_remaining>=0)
        assert(t2_remaining>=0)
        assert((t1_remaining>0) | (t2_remaining>0))

        # only select the allocation for which there are enough resources remaining 
        temp_df = main_df[(d_t1<=t1_remaining) & \
            (d_t2<=t2_remaining) & ((main_df["{}_t1".format(self.mt_type)]>0) | (main_df["{}_t2".format(self.mt_type)]>0)) & \
            (main_df["c"]>self.cost_array[workload_index])].copy()


        if len(temp_df):
            """ This means that there is an entry that can be allocated. """

            # Now check if there is a previous allocation 
            cur_entry = self.allocation_entry_list[workload_index]
            if cur_entry is not None:
                """ This means they had a previous allocation. """

                # Previous cumulative latency reduced 
                prev_d_lat = cur_entry["cum_lat_reduced_{}".format(self.mt_type)].item()

                # Subtract the latency reduced from the main table and adjust the cum_lat_reduced_wb 
                temp_df.loc[:, "cum_lat_reduced_{}".format(self.mt_type)]  = temp_df["cum_lat_reduced_{}".format(self.mt_type)]-prev_d_lat
                temp_df.loc[:, "cum_lat_reduced_{}_per_dollar".format(self.mt_type)] = temp_df.loc[:, "cum_lat_reduced_{}".format(self.mt_type)]/d_c

            metric = temp_df["cum_lat_reduced_{}_per_dollar".format(self.mt_type)].max()
            metric_entry = temp_df[temp_df["cum_lat_reduced_{}_per_dollar".format(self.mt_type)] == metric].iloc[0]
            self.metric_array[workload_index] = metric 
            self.metric_entry_list[workload_index] = metric_entry
        else:
            """ No allocations found for the amount of resources remaining in the cache """

            self.metric_array[workload_index] = -1 
            self.metric_entry_list[workload_index] = None


    def allocate(self):
        """ Find the workload with the maximum metric and allocate the cache resources to it. 
        """

        workload_index = np.argmax(self.metric_array)
        workload_name = self.workload_name_list[workload_index]
        df = self.data[self.workload_name_list[workload_index]]
        allocation_entry = self.metric_entry_list[workload_index]
        

        if allocation_entry is not None:
            """ There has been a previous allocation so update how much additional resources is consumed. """
            t1_increase = allocation_entry["{}_t1".format(self.mt_type)].item() - self.allocation_array[workload_index][0]
            t2_increase = allocation_entry["{}_t2".format(self.mt_type)].item() - self.allocation_array[workload_index][1]
            cost_increase = allocation_entry["c"].item() - self.cost_array[workload_index]
            self.allocation_array[workload_index][0] += t1_increase
            self.allocation_array[workload_index][1] += t2_increase
            self.cost_array[workload_index] += cost_increase
            self.allocation_entry_list[workload_index] = allocation_entry

            remaining_sizes = self.get_remaining_size()

            log_entry = "{},{},{},{},{},{},{}".format(workload_name, t1_increase, t2_increase, \
                cost_increase, self.metric_array[workload_index], remaining_sizes[0], remaining_sizes[1])

            print(log_entry)
            self.log_path_handle.write("{}\n".format(log_entry))

        return allocation_entry


    def get_latency(self, summary_file):
        summary_handle = summary_file.open("w+")
        mt_cache = MTCache()
        total_lat = 0.0
        total_cost = 0.0
        for workload_index in range(len(self.workload_name_list)):
            workload_name = self.workload_name_list[workload_index]
            allocation_entry = self.allocation_entry_list[workload_index]

            if allocation_entry is None:
                allocation_entry = self.data[self.workload_name_list[workload_index]]["df"].iloc[0]

            tier_1_size = allocation_entry["{}_t1".format(self.mt_type)].item()
            tier_2_size = allocation_entry["{}_t2".format(self.mt_type)].item()
            cost = allocation_entry["c"].item()
            mean_lat = allocation_entry["{}_min_lat".format(self.mt_type)].item()
            lat_reduced = allocation_entry["lat_reduced_{}".format(self.mt_type)].item()

            workload_string  = "{},{},{},{},{}".format(workload_name, 
                tier_1_size,
                tier_2_size, 
                cost,
                mean_lat, 
                lat_reduced)

            print(workload_string)
            summary_handle.write("{}\n".format(workload_string))

            total_lat += mean_lat 
            total_cost += cost 
        
        remaining_sizes = self.get_remaining_size()
        overall_string = "Overall,{},{},{},{},{}".format(remaining_sizes[0], remaining_sizes[1], 
            total_cost, total_lat/len(self.workload_name_list), total_lat/total_cost)
        print(overall_string)
        summary_handle.write("{}\n".format(overall_string))
        summary_handle.close()

