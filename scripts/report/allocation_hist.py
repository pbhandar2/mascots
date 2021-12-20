import pathlib, json 
import pandas as pd 
from collections import namedtuple, defaultdict  
import matplotlib.pyplot as plt

MT_TYPE="wb"
COST_DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/cost")
LAT_DATA_DIR = pathlib.Path("../cache_allocater/greedy_lat_data")
MT_ENTRY = namedtuple('MTEntry', ['workload', 't1', 't2', 'cost', 'lat'])
OVERALL_ENTRY = namedtuple('MTEntry', ['workload', 't1', 't2', 'cost', 'lat', 'lat_dollar'])


def get_lat_without_cache(workload_name, mt_label):
    workload_cost_data_dir = COST_DATA_DIR.joinpath(workload_name)
    data_path = workload_cost_data_dir.joinpath("{}.csv".format(mt_label))
    df = pd.read_csv(data_path)
    entry = df[(df["{}_t1".format(MT_TYPE)]==0) & (df["{}_t2".format(MT_TYPE)] == 0)].iloc[0]
    return entry["{}_min_lat".format(MT_TYPE)]


def get_base_latency(mt_label):
    data = {}
    for i in range(1,107):
        if i<10:
            workload_name = "w0{}".format(i)
        else:
            workload_name = "w{}".format(i)
        data[workload_name] = get_lat_without_cache(workload_name, mt_label)
    return data 

def init_per_workload_dict():
    data = {}
    for i in range(1,107):
        if i<10:
            workload_name = "w0{}".format(i)
        else:
            workload_name = "w{}".format(i)
        data[workload_name] = None 
    return data 


def get_base_latency_all_devices():
    base_lat_data = {}
    for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
        with config_path.open("r") as f:
            device_config = json.load(f)
            mt_label = "_".join([_["label"] for _ in device_config])
            base_lat_data[mt_label] = get_base_latency(mt_label)
    return base_lat_data 
BASE_LAT_DATA = get_base_latency_all_devices()


def read_summary_data(path, mt_label):
    lat_reduced_data = init_per_workload_dict()
    with path.open("r") as f:
        line = f.readline() 
        while line:
            split_line = line.split(",")
            wname, t1, t2, cost, lat = split_line[0], float(split_line[1]), \
                float(split_line[2]), float(split_line[3]), float(split_line[4])
            
            if "Overall" not in line:
                lat_reduced = BASE_LAT_DATA[mt_label][wname] - lat 
                lat_reduced_data[wname] = {
                    "t1": t1,
                    "t2": t2,
                    "lat_reduced": lat_reduced 
                }
                entry = MT_ENTRY(wname, t1, t2, cost, lat)
            else:
                lat_per_dollar = float(split_line[5])
                entry = OVERALL_ENTRY(wname, t1, t2, cost, lat, lat_per_dollar)

            line = f.readline() 
    return lat_reduced_data 


per_storage_device_lat_vs_cost = defaultdict(list)
for lat_data_file in LAT_DATA_DIR.iterdir():
    if "summary" not in str(lat_data_file):
        continue 

    current_mt_label = "_".join([str(_) for _ in lat_data_file.name.split("_")[:3]])
    storage_label = current_mt_label.split("_")[-1]
    mt_label = "_".join([str(_) for _ in lat_data_file.name.split("_")[:2]])

    cur_data = read_summary_data(lat_data_file, current_mt_label)
    per_storage_device_lat_vs_cost[storage_label].append({
        mt_label: cur_data 
    })


for storage_entry in per_storage_device_lat_vs_cost:
    plt.figure(figsize=[24, 6])
    plt.rcParams.update({'font.size': 25})

    per_storage_data = per_storage_device_lat_vs_cost[storage_entry]
    num_config = len(cur_data)+1

    first_data = per_storage_data[0]
    config_list = list(first_data.keys())
    print(config_list)
    data = first_data[config_list[0]]
    
    label_array = []
    t1_list = []
    t2_list = []
    for i in range(1, len(data)+1):
        if i<10:
            workload_name = "w0{}".format(i)
        else:
            workload_name = "w{}".format(i)
        t1_list.append(data[workload_name]["t1"])
        t2_list.append(data[workload_name]["t2"])
        label_array.append(workload_name)

    print(label_array)
    print(t1_list)
    print(t2_list)

    ax = plt.subplot(1,1,1)
    width = 1
    ax.bar(label_array, t1_list, width)
    ax.bar(label_array, t2_list, width, bottom=t1_list)
    ax.set_xticks([], [])
    ax.set_yscale("symlog")
    
    plt.tight_layout()
    output_path = "{}.png".format(storage_entry)
    plt.savefig(output_path)
    plt.close()


        

