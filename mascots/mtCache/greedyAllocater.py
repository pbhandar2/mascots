import pathlib 
import numpy as np 
import pandas as pd 


class GreedyAllocater:

    def __init__(self, workload_name_list, device_config, size_array):
        self.data_dir = pathlib.Path("/research2/mtc/cp_traces/mascots/cost") 
        self.workload_name_list = workload_name_list 
        self.num_workloads = len(workload_name_list)
        self.device_config = device_config
        self.mt_label = "_".join([_["label"] for _ in self.device_config])

        self.size_array = size_array
        self.allocation_array = np.zeros(shape=(self.num_workloads, 2), dtype=int)
        self.metric_array = np.zeros(self.num_workloads, dtype=float)
        self.cost_array = np.zeros(self.num_workloads, dtype=int)

        # list contains Series or int 
        self.metric_entry_list = [None]*self.num_workloads

        """ dictionary contains DataFrame of each workload's exclusive cache performance for different cost """
        self.data = {}

    
    def run(self):
        # load the data and metric of each workload 
        self.load_data()
        for index in range(len(self.workload_name_list)):
            self.load_metric(index)

        """ While we have an allocation that fits the resources remaining, keep allocating. 
        """
        allocation_entry = self.allocate()
        while allocation_entry is not None:
            for index in range(len(self.workload_name_list)):
                self.load_metric(index)
            allocation_entry = self.allocate()

        
    def get_remaining_size(self):
        total_allocation = np.sum(self.allocation_array, axis=0)
        return [self.size_array[0]-total_allocation[0], self.size_array[1]-total_allocation[1]]


    def allocate(self):
        """ Find the workload with the maximum metric and allocate the cache resources to it. 
        """

        workload_index = np.argmax(self.metric_array)
        workload_name = self.workload_name_list[workload_index]
        df = self.data[self.workload_name_list[workload_index]]
        allocation_entry = self.metric_entry_list[workload_index]

        if allocation_entry is not None:
            """ There has been a previous allocation so update how much additional resources is consumed. """
            t1_increase = allocation_entry["wb_t1"].item() - self.allocation_array[workload_index][0]
            t2_increase = allocation_entry["wb_t2"].item() - self.allocation_array[workload_index][1]
            cost_increase = allocation_entry["c"].item() - self.cost_array[workload_index]
            self.allocation_array[workload_index][0] += t1_increase
            self.allocation_array[workload_index][1] += t2_increase
            self.cost_array[workload_index] += cost_increase

        return allocation_entry

        
    def load_metric(self, workload_index):
        """ Load the value of the metric to be maximized for a given workload 
        """

        # The amount of additional T1, T2 and cost for each cache allocation
        main_df = self.data[self.workload_name_list[workload_index]]["df"].copy()
        d_t1 = main_df["wb_t1"] - self.allocation_array[workload_index][0]
        d_t2 = main_df["wb_t2"] - self.allocation_array[workload_index][1]
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
            (d_t2<=t2_remaining) & ((main_df["wb_t1"]>0) | (main_df["wb_t2"]>0)) & \
            (main_df["c"]>self.cost_array[workload_index])].copy()


        if len(temp_df):
            """ This means that there is an entry that can be allocated. """

            # Now check if there is a previous allocation 
            cur_entry = self.metric_entry_list[workload_index]
            if cur_entry is not None:
                """ This means they had a previous allocation. """

                # Previous cumulative latency reduced 
                prev_d_lat = cur_entry["cum_lat_reduced_wb"].item()

                # Subtract the latency reduced from the main table and adjust the cum_lat_reduced_wb 
                temp_df.loc[:, "cum_lat_reduced_wb"]  = temp_df["cum_lat_reduced_wb"]-prev_d_lat
                temp_df.loc[:, "cum_lat_reduced_wb_per_dollar"] = temp_df.loc[:, "cum_lat_reduced_wb"]/d_c

            metric = temp_df["cum_lat_reduced_wb_per_dollar"].max()
            metric_entry = temp_df[temp_df["cum_lat_reduced_wb_per_dollar"] == metric].iloc[0]
            self.metric_array[workload_index] = metric 
            self.metric_entry_list[workload_index] = metric_entry
        else:
            """ No allocations found for the amount of resources remaining in the cache """

            self.metric_array[workload_index] = -1 
            self.metric_entry_list[workload_index] = None


    def load_data(self):
        """ Load the DataFrame of each workload and add the necessary columns 
        """

        for workload_index, workload_name in enumerate(self.workload_name_list):
            data_path = self.data_dir.joinpath(workload_name, "{}.csv".format(self.mt_label))
            df = pd.read_csv(data_path)
            df["lat_reduced_wb"] = df["wb_min_lat"].shift(1, fill_value=0)-df["wb_min_lat"]
            df["cum_lat_reduced_wb"] = df["lat_reduced_wb"].cumsum()
            df["cum_lat_reduced_wb_per_dollar"] = df["cum_lat_reduced_wb"]/df["c"]
            self.data[workload_name] = {
                "df": df
            }


    def get_info(self):
        """ For each workload, generate how much cache is allocated and what the expected performance is. 
        """
        
        mean_lat_array = []
        for workload_index, workload_name in enumerate(self.workload_name_list):
            allocation_entry = self.metric_entry_list[workload_index]
            cost_value = int(self.cost_array[workload_index])

            df = self.data[workload_name]["df"]
            mean_lat = df[df["c"] == cost_value]["wb_min_lat"].item()
            mean_lat_array.append(mean_lat)

        print("Mean Latency: {}".format(np.mean(mean_lat_array)))


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
        overall_string = "Overall,{},{},{},{}".format(remaining_sizes[0], remaining_sizes[1], total_cost, total_lat/len(self.workload_name_list))
        summary_handle.write("{}\n".format(overall_string))
        summary_handle.close()
