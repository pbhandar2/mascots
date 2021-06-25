import pathlib 
import numpy as np 
import pandas as pd 


class MTCacheAllocater:

    def __init__(self, workload_name_list, device_list, size_list, metric_name):
        self.data_dir = pathlib.Path("/research2/mtc/cp_traces/mascots/exclusive")
        self.workload_name_list = workload_name_list 
        self.device_list = device_list 
        self.size_list = size_list 
        self.metric_name = metric_name 

        self.data = {}
        self.t1_allocation_array = np.zeros(len(self.workload_name_list), dtype=int)
        self.metric_array = np.zeros(len(self.workload_name_list), dtype=float)


    def run(self):
        self.load_data()
        max_metric_index = np.argmax(self.metric_name)
        for i in range(100):
            self.allocate(max_metric_index)


    def allocate(self, workload_index):
        self.t1_allocation_array[workload_index] += 1
        self.metric_array[workload_index] = self.get_metric(workload_index)
        print(self.t1_allocation_array)
        print(self.metric_array)
        


    def get_metric(self, workload_index):
        workload_name = self.workload_name_list[workload_index]
        df = self.data[workload_name]["df"]
        workload_index = self.data[workload_name]["index"]
        entry = df.iloc[self.t1_allocation_array[workload_index]][self.metric_name]
        return entry if entry > 0 else -1 


    def load_data(self):
        cur_cache_size = 1 
        cache_label = "_".join([_["label"] for _ in self.device_list])
        for workload_index, workload_name in enumerate(self.workload_name_list):
            workload_data_path = self.data_dir.joinpath(self.data_dir, workload_name, cache_label, "{}.csv".format(cur_cache_size))
            self.data[workload_name] = {
                "df": pd.read_csv(workload_data_path),
                "index": workload_index
            }
            self.metric_array[workload_index] = self.get_metric(workload_index)






            





    












    #     self.workloads = workload_array
    #     self.devices = device_array
    #     self.sizes = size_array
    #     self.sizes_remaining = np.array(size_array, copy=True)
    #     self.metric = metric_name
    #     self.device_config_name = "_".join([self.devices[i]["label"] for i in range(3)])
    #     self.data = {}

    #     self.load_data()


    #     # self.config = config 
    #     # self.tier_size_array = tier_size_array
    #     # self.tier_size_remaining = np.array(tier_size_array, copy=True)
    #     # self.metric_name = metric_name
    #     # self.device_config_name = "{}_{}_{}".format(self.config[0]["label"], self.config[1]["label"], self.config[2]["label"])
    #     # self.data = {}
    #     # self.metric_array = np.zeros(len(self.workload_file_list), dtype=float)
    #     # self.workload_name_array = [None] * len(self.workload_file_list)
    #     # self.allocation = {}
    #     # self.load_data()
    #     # self.main()


    # def run(self, cache_name):
    #     cache_data_dir = self.data_dir("/research2/mtc/cp_traces/mascots/exclusive/{}/{}")


    # def load_data(self):
    #     for workload_index, workload_name in enumerate(self.workloads):
    #         workload_file_path = self.data_dir.joinpath(workload_name, "{}.csv".format(self.device_config_name))
    #         self.data[workload_name] = {
    #             "df": pd.read_csv(workload_file_path,
    #                 names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", \
    #                     "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])
    #         }
    #         assert(len(self.data[workload_name]["df"])>0)

    #         self.data[workload_name]["df"]["wb_d_lat_abs"] = self.data[workload_name]["df"]["wb_d_lat"] \
    #             - self.data[workload_name]["df"]["wb_d_lat"].shift(1, fill_value=0.0)


    #         print(self.data[workload_name]["df"])

    






    # def get_metric(self, workload_name, index):
    #     df = self.data[workload_name]["df"]
    #     self.data[workload_name]["df"] = df[(df["wb_t1"]<=self.tier_size_remaining[0]) & (df["wb_t2"]<=self.tier_size_remaining[1])]
    #     return self.data.iloc[index][self.metric_name]


    # def allocate_cache(self, max_entry_index):
    #     workload_name = self.workload_name_array[max_entry_index]
    #     df = self.data[workload_name]["df"]

    #     if workload_name not in self.allocation:
    #         allocation_entry = df.iloc[0]
    #         self.tier_size_remaining[0] -= allocation_entry["wb_t1"]
    #         self.tier_size_remaining[1] -= allocation_entry["wb_t2"]
    #     else:
    #         prev_allocation_entry = self.allocation[workload_name]
    #         allocation_entry = df.iloc[int(self.allocation[workload_name]["c"])]
    #         self.tier_size_remaining[0] -= allocation_entry["wb_t1"] - prev_allocation_entry["t1"]
    #         self.tier_size_remaining[1] -= allocation_entry["wb_t2"] - prev_allocation_entry["t2"]
        
    #     self.allocation[workload_name] = {
    #         "t1": allocation_entry["wb_t1"],
    #         "t2": allocation_entry["wb_t2"],
    #         "c": allocation_entry["c"]
    #     }

    #     self.data[workload_name]["df"] = df[(df["wb_t1"]<=self.tier_size_remaining[0]) & (df["wb_t2"]<=self.tier_size_remaining[1])]

    #     self.metric_array[max_entry_index] = self.get_max_metric(workload_name)


    # # def load_data(self):

    # #     workload_name_array = []
    # #     for workload_index, workload_name in enumerate(self.workload_file_list):
    # #         workload_file_path = self.cost_data_dir.joinpath(workload_name, "{}.csv".format(self.device_config_name))
    # #         self.workload_name_array[workload_index] = workload_name
    # #         print("Loading {}".format(workload_name))
    # #         self.data[workload_name] = {
    # #             "df": pd.read_csv(workload_file_path,
    # #                 names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])
    # #         }
    # #         assert(len(self.data[workload_name]["df"])>0)
    # #         self.data[workload_name]["df"]["wb_d_lat_abs"] = self.data[workload_name]["df"]["wb_d_lat"]-self.data[workload_name]["df"]["wb_d_lat"].shift(1, fill_value=0.0)
    # #         self.metric_array[workload_index] = self.get_max_metric(workload_name)
        
        
    # #     print("Data Loaded!")


    # def main(self):
    #     for i in range(10):
    #         print("remaining size ", self.tier_size_remaining)
    #         max_entry_index = np.argmax(self.metric_array)
    #         self.allocate_cache(max_entry_index)
    #         print("remaining size ", self.tier_size_remaining)
    #         print(self.allocation)



