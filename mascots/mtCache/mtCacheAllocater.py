import pathlib 
import numpy as np 
import pandas as pd 


class MTCacheAllocater:

    def __init__(self, workload_name_list, device_list, size_list, metric_name, log_file=pathlib.Path("test.csv")):
        self.data_dir = pathlib.Path("/research2/mtc/cp_traces/mascots/exclusive")
        self.workload_name_list = workload_name_list 
        self.num_workloads = len(workload_name_list)
        self.device_list = device_list 
        self.size_list = size_list 
        self.metric_name = metric_name 
        self.mt_label = "_".join([_["label"] for _ in self.device_list])

        self.allocation_array = np.zeros(shape=(self.num_workloads, 2), dtype=int)

        self.t1_data = {}
        self.t2_data = {}
        self.t1_allocation_array = np.zeros(len(self.workload_name_list), dtype=int)
        self.t2_allocation_array = np.zeros(len(self.workload_name_list), dtype=int)
        self.t1_lat_reduced_array = np.zeros(len(self.workload_name_list), dtype=float)
        self.t2_lat_reduced_array = np.zeros(len(self.workload_name_list), dtype=float)
        self.cost_array = np.zeros(len(self.workload_name_list), dtype=float)
        self.metric_array = [None]*len(self.workload_name_list)
        self.t2_metric_array = [None]*len(self.workload_name_list)

        self.log_handle = log_file.open("w+")


    def run(self):
        self.load_data()
        t1_remaining = self.get_t1_remaining()
        print("T1 Remaining: {}".format(t1_remaining))
        while t1_remaining > 0:
            print("T1 Remaining: {}".format(t1_remaining))
            self.allocate_t1()
            t1_remaining = self.get_t1_remaining()
        print("Done with T1!")

        
        self.load_t2_data()
        # t2_remaining = self.get_t2_remaining()
        # size_allocated = self.allocate_t2()
        # print("T2 Remaining {} Size Alocated {}".format(t2_remaining, size_allocated))
        # while ((t2_remaining > 0) and (size_allocated>0)):
        #     print("T2 Remaining: {}".format(t2_remaining))
        #     size_allocated = self.allocate_t2()
        #     t2_remaining = self.get_t2_remaining()
        #     print("Size Allocated: {}".format(size_allocated))

        # print(self.allocation_array)


    def allocate_t1(self):
        workload_index = np.argmax([_["lat_red_per_dollar"].item() for _ in self.metric_array])
        workload_name = self.workload_name_list[workload_index]
        allocation_entry = self.metric_array[workload_index]
        self.allocation_array[workload_index][0] += allocation_entry["t1"].item()
        self.t1_lat_reduced_array[workload_index] += allocation_entry["d_lat_wb"].item()
        self.cost_array[workload_index] += allocation_entry["cost"].item()
        self.update_t1_data(workload_index)

        print("Allocated T1 {} to {}".format(allocation_entry["t1"].item(), workload_name))

        total_cost = np.sum(self.cost_array)
        total_latency_reduced = np.sum(self.t1_lat_reduced_array)
        total_t1_size = np.sum(self.allocation_array[:,0])
        log_string = "{},{},0,{},{},{}\n".format(workload_name, total_t1_size, total_cost, total_latency_reduced, total_latency_reduced/total_cost)
        self.log_handle.write(log_string)

        for _ in range(len(self.workload_name_list)):   
            self.update_metric(_)


    def get_max_metric_and_workload_index(self, tier_number):
        max_metric_workload_index = -1
        max_metric = -1 
        metric_entry = None 

        if tier_number == 1:
            metric_array = self.t2_metric_array
        elif tier_number == 2:
            metric_array = self.t2_metric_array

        for index, metric in enumerate(metric_array):
            if type(metric) != int:
                if metric["d_lat_wb"].item() > max_metric and metric["d_lat_wb"].item()>0:
                    max_metric = metric["d_lat_wb"].item()
                    max_metric_workload_index = index 
                    metric_entry = metric

        return max_metric_workload_index, metric_entry


    def allocate_t2(self):
        workload_index, allocation_entry = self.get_max_metric_and_workload_index(2)
        if workload_index < 0:
            return -1 
        
        workload_name = self.workload_name_list[workload_index]
        self.allocation_array[workload_index][1] += allocation_entry["t2"].item()
        self.t2_lat_reduced_array[workload_index] += allocation_entry["d_lat_wb"].item()
        self.cost_array[workload_index] += allocation_entry["cost"].item()
        self.update_t2_data(workload_index)

        # print("Current allocation: {}".format(self.allocation_array[workload_index][1]))
        print("Allocated T2 {} to {} with dlat {}".format(
            allocation_entry["t2"].item(), 
            workload_name, 
            allocation_entry["d_lat_wb"].item()))
        # print(allocation_entry)

        total_cost = np.sum(self.cost_array)
        total_latency_reduced = np.sum(self.t1_lat_reduced_array)
        total_t1_size = np.sum(self.allocation_array[:,0])
        total_t2_size = np.sum(self.allocation_array[:,1])
        log_string = "{},{},{},{},{},{}\n".format(workload_name, total_t1_size, total_t2_size, total_cost, total_latency_reduced, total_latency_reduced/total_cost)
        self.log_handle.write(log_string)

        for _ in range(len(self.workload_name_list)):   
            self.update_t2_metric(_)

        return allocation_entry["t2"].item()


    def load_data(self):
        for workload_index, workload_name in enumerate(self.workload_name_list):
            self.update_t1_data(workload_index)
            self.update_metric(workload_index)


    def load_t2_data(self):
        for workload_index, workload_name in enumerate(self.workload_name_list):
            self.update_t2_data(workload_index)
            self.update_t2_metric(workload_index)


    def get_t1_remaining(self):
        total_t1_allocation = np.sum(self.allocation_array[:,0])
        t1_remaining = self.size_list[0] - total_t1_allocation
        return t1_remaining


    def get_t2_remaining(self):
        total_t2_allocation = np.sum(self.allocation_array[:,1])
        t2_remaining = self.size_list[1] - total_t2_allocation
        return t2_remaining


    def update_metric(self, workload_index):
        workload_name = self.workload_name_list[workload_index]
        t1_remaining = self.get_t1_remaining()
        final_df = self.t1_data[workload_name]["df"]
        final_df = final_df[final_df["t1"]<=t1_remaining]
        max_metric = -1
        if len(final_df):
            entry = final_df[final_df["lat_red_per_dollar"] == final_df["lat_red_per_dollar"].max()]
            max_metric = entry["lat_red_per_dollar"].item()

            if max_metric > 0:
                self.metric_array[workload_index] = entry 
            else:
                self.metric_array[workload_index] = -1 


    def update_t2_metric(self, workload_index):
        workload_name = self.workload_name_list[workload_index]
        t2_remaining = self.get_t2_remaining()
        final_df = self.t2_data[workload_name]["df"]
        final_df = final_df[final_df["t2"]<=t2_remaining]
        max_metric = -1
        if len(final_df):
            entry = final_df[final_df["lat_red_per_dollar"] == final_df["lat_red_per_dollar"].max()]
            max_metric = entry["lat_red_per_dollar"].item()

            if max_metric > 0:
                self.t2_metric_array[workload_index] = entry
            else:
                self.t2_metric_array[workload_index] = -1

    
    def update_t1_data(self, workload_index, chunksize=100):
        workload_name = self.workload_name_list[workload_index]
        t1_remaining = self.get_t1_remaining()
        cur_t1_allocation = self.allocation_array[workload_index][0]
        chunksize = min(chunksize, t1_remaining)
        
        output_array = []
        for t1_size in range(cur_t1_allocation+1, cur_t1_allocation+1+chunksize):
            workload_data_path = self.data_dir.joinpath(self.data_dir, workload_name, self.mt_label, "{}.csv".format(t1_size))
            if workload_data_path.is_file():
                df = pd.read_csv(workload_data_path)
                output_array.append(df.iloc[0])

        final_df = pd.DataFrame(output_array)
        if len(final_df) > 0:
            final_df["t1"] -= cur_t1_allocation
            final_df["d_lat_wb"] -= self.t1_lat_reduced_array[workload_index]
            final_df["cost"] -= self.cost_array[workload_index]
            final_df["lat_red_per_dollar"] = final_df["d_lat_wb"]/final_df["cost"]
            final_df = final_df[final_df["t1"]<=t1_remaining]
            self.t1_data[workload_name] = {
                "df": final_df
            }


    def update_t2_data(self, workload_index, chunksize=100):
        workload_name = self.workload_name_list[workload_index]
        t2_remaining = self.get_t2_remaining()
        cur_t2_allocation = self.t2_allocation_array[workload_index]
        chunksize = min(chunksize, t2_remaining)
        t1_size = self.allocation_array[workload_index][0]

        workload_data_path = self.data_dir.joinpath(self.data_dir, workload_name, self.mt_label, "{}.csv".format(t1_size))
        final_df = pd.read_csv(workload_data_path)
        final_df = final_df[final_df["t2"]>0.0]
        if len(final_df) > 0:
            final_df["t2"] -= cur_t2_allocation
            final_df["d_lat_wb"] -= (self.t2_lat_reduced_array[workload_index] + self.t1_lat_reduced_array[workload_index])
            final_df["cost"] -= self.cost_array[workload_index]
            final_df["lat_red_per_dollar"] = final_df["d_lat_wb"]/final_df["cost"]
            final_df = final_df[final_df["t2"]<=t2_remaining]
            self.t2_data[workload_name] = {
                "df": final_df
            }


    
