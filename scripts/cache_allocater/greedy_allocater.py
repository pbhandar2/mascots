import argparse 
import pathlib 
import json 

from mascots.mtCache.mtCacheAllocater import MTCacheAllocater

from mascots.mtCache.greedyAllocater import GreedyAllocater

device_config_name = "D2_S1_H1"
device_config_path = "../../mascots/mtCache/device_config/{}.json".format(device_config_name)
with open(device_config_path) as f:
    device_config = json.load(f)

workloads = []

t1_device = int((device_config[0]["size"]*1024)/10)
t2_device = int((device_config[1]["size"]*1024)/10)
size_array = [t1_device, t2_device]
for _ in range(100,107):
    workload_name = "w{}".format(_)
    if _ < 10:
        workload_name = "w0{}".format(_)
    workloads.append(workload_name)

allocation_log_file = "{}_{}_{}.csv".format(size_array[0], size_array[1], device_config_name)
# allocater = MTCacheAllocater(workloads, device_config, size_array, "d_lat_per_dollar")
# allocater.run()

allocater = GreedyAllocater(workloads, device_config, size_array)
allocater.run()
allocater.get_info()


