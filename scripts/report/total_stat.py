import pathlib, json 
import numpy as np 
import pandas as pd 
from collections import defaultdict, OrderedDict
from functools import partial
from matplotlib.ticker import FormatStrFormatter
from mascots.mtCache.mtCache import MTCache

import matplotlib.pyplot as plt

def load_cost_per_block(df, df_label, total_ws):
    out_array = []
    cost_per_gb = 1024*256*df["c"]/total_ws 
    for index, item in cost_per_gb.items():
        out_array.append([item, df_label])
    return out_array 


mt_type = "wt"
DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/cost")
WORKLOAD_DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/general/block_read_write_stats.csv")
WORKLOAD_DF = pd.read_csv(WORKLOAD_DATA_DIR)

DEVICE_LIST = []
T1_LAT_DIFF = {}
T2_LAT_DIFF = {}
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        device_config = json.load(f)
        mt_label = "_".join([_["label"] for _ in device_config])
        DEVICE_LIST.append(device_config)

        cur_t1_lat_diff = (device_config[1]["read_lat"]+device_config[1]["write_lat"]) - (device_config[0]["read_lat"]+device_config[0]["write_lat"])
        cur_t2_lat_diff = (device_config[-1]["read_lat"]+device_config[-1]["write_lat"]) - (device_config[1]["read_lat"]+device_config[1]["write_lat"])

        mt_cache = MTCache()
        wb_lat = mt_cache.get_exclusive_tier_lat_wb([1,2], device_config)
        wb_st_lat = mt_cache.get_single_tier_lat_wb([1,0], device_config)

        T1_LAT_DIFF[mt_label] = wb_lat 
        T2_LAT_DIFF[mt_label] = cur_t2_lat_diff/(device_config[1]["read_lat"]+device_config[1]["write_lat"])



total_data_points = 0 
total_st_t1_data_points = 0 
total_st_t2_data_points = 0 
total_st_data_points = 0 

per_device_count = defaultdict(partial(np.zeros, 4))
dollar_per_block_vs_cache_type_data = defaultdict(list)
dollar_per_block_list = []
mt_p_vs_np_data = defaultdict(list)
mt_p_vs_np_lat_data = defaultdict(list)


mt_np_vs_st_data = defaultdict(list)


for workload_dir in DATA_DIR.iterdir():
    workload_name = workload_dir.name 
    total_ws = WORKLOAD_DF[WORKLOAD_DF["workload"]==workload_name]["total_ws"].item()

    for device_file_path in workload_dir.iterdir():
        device_config_label = device_file_path.stem

        df = pd.read_csv(device_file_path)
        
        
        mt_non_pyramid_df = df[(df["{}_t1".format(mt_type)]>=df["{}_t2".format(mt_type)]) & (df["{}_t2".format(mt_type)]>0)]
        mt_pyramid_df = df[(df["{}_t1".format(mt_type)]<df["{}_t2".format(mt_type)]) & (df["{}_t1".format(mt_type)]>0)]
        st_t1_df = df[(df["{}_t2".format(mt_type)]<1) & (df["{}_t1".format(mt_type)]>0)]
        st_t2_df = df[(df["{}_t1".format(mt_type)]<1) & (df["{}_t2".format(mt_type)]>0)]
        device_data_points = len(mt_non_pyramid_df)+len(mt_pyramid_df)+len(st_t1_df)+len(st_t2_df)
        total_data_points += device_data_points
        total_st_t1_data_points += len(st_t1_df)
        total_st_t2_data_points += len(st_t2_df)
        total_st_data_points += len(st_t1_df) + len(st_t2_df)

        # there is one entry t1=0, t2=0 which does not belong to any caching cateogory so wouldn't be in any DFs
        assert(len(df) == device_data_points+1)

        # collect the count per device combination 
        per_device_count[device_config_label][0] += len(st_t1_df)
        per_device_count[device_config_label][1] += len(st_t2_df)
        per_device_count[device_config_label][2] += len(mt_pyramid_df)
        per_device_count[device_config_label][3] += len(mt_non_pyramid_df)

        # collect info per cost scaled by working set size 
        dollar_per_block_vs_cache_type_data["ST-T1"] += load_cost_per_block(st_t1_df, "ST-T1", total_ws)
        dollar_per_block_vs_cache_type_data["ST-T2"] += load_cost_per_block(st_t2_df, "ST-T2", total_ws)
        dollar_per_block_vs_cache_type_data["MT-P"] += load_cost_per_block(mt_pyramid_df, "MT-P", total_ws)
        dollar_per_block_vs_cache_type_data["MT-NP"] += load_cost_per_block(mt_non_pyramid_df, "MT-NP", total_ws)

        if len(mt_pyramid_df) > 0 or len(mt_non_pyramid_df) > 0:
            # percentage diff in latency 
            diff_non_pyramid = 100*(mt_non_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))] \
                - mt_non_pyramid_df.loc[:, ("{}_min_lat".format(mt_type))])/mt_non_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))]
            non_pyramid_stats = diff_non_pyramid.describe()

            diff_pyramid = 100*(mt_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))] \
                - mt_pyramid_df.loc[:, ("{}_min_lat".format(mt_type))])/mt_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))]
            pyramid_stats = diff_pyramid.describe()

            if len(mt_non_pyramid_df):
                mt_p_vs_np_lat_data[device_config_label].append(non_pyramid_stats["mean"])

            try:
                mt_p_vs_np_data[device_config_label].append(100*len(mt_non_pyramid_df)/(len(mt_non_pyramid_df)+len(mt_pyramid_df)))
            except ZeroDivisionError:
                pass







print("\n\nBASE\n")
print("Total Data Points: {}".format(total_data_points))
print("Total ST: {}, {}%".format(total_st_data_points, 100*total_st_data_points/total_data_points))
print("Total ST_T1: {}, {}%".format(total_st_t1_data_points, 100*total_st_t1_data_points/total_data_points))
print("Total ST_T2: {}, {}%".format(total_st_t2_data_points, 100*total_st_t2_data_points/total_data_points))
print("ST_T2 - ST_T1: {}, {}%".format(total_st_t2_data_points-total_st_t1_data_points,
    100*total_st_t2_data_points/(total_st_t2_data_points+total_st_t1_data_points)))



print("\n\nPER DEVICE\n")
label_array = []
st_vs_mt_data = []
for device_config_label in per_device_count:
    data = 100*per_device_count[device_config_label]/np.sum(per_device_count[device_config_label])
    label_array.append(device_config_label)
    st_vs_mt_data.append(np.array([data[0], data[1], data[2], data[3]]))
    print("{},{},{},{},{}".format(device_config_label, data[0], data[1], data[2], data[3]))


st_vs_mt_data = np.array(st_vs_mt_data)
mt_count = st_vs_mt_data[:,2]+st_vs_mt_data[:,3]
index_array = np.argsort(mt_count, axis=0)

print("MAX MT Optimal (%): {} {}".format(label_array[index_array[-1]], mt_count[index_array[-1]]))
print("Min MT Optimal (%): {} {}".format(label_array[index_array[0]], mt_count[index_array[0]]))

plt.figure(figsize=[24, 6])
plt.rcParams.update({'font.size': 25})
ax = plt.subplot(1,1,1)

mt_st_percentage_csv = pathlib.Path("mt_st_percentage_{}.csv".format(mt_type)).open("w+")

x = []
hatch_array = [".", "+", "*", "o"]
for index, i in enumerate(index_array):

    data = st_vs_mt_data[i]
    max_val = index*3
    x.append(max_val-0.5)

    ax.bar([max_val-1], data[0], hatch=hatch_array[0], color=(0.2, 0.4, 0.6, 0.6), label="ST-T1")
    ax.bar([max_val-1], data[1], bottom=data[0], hatch=hatch_array[1], color=(0.2, 0.4, 0.6, 0.6), label="ST-T2")

    ax.bar([max_val], data[2], hatch=hatch_array[2], color=(0.2, 0.4, 0.6, 0.6), label="MT-P")
    ax.bar([max_val], data[3], bottom=data[2], hatch=hatch_array[3], color=(0.2, 0.4, 0.6, 0.6), label="MT-NP")

    mt_st_percentage_csv.write("{},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(label_array[i], 
        data[0], data[1], data[2], data[3]))

mt_st_percentage_csv.close()

ax.set_xticks(x)
ax.set_xticklabels(label_array)
plt.ylabel("Optimal Configuration (%)")
plt.xlabel("Device Combinations")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), 
    bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.)


plt.tight_layout()
plt.savefig('device_hist_{}.png'.format(mt_type))
plt.close()




plt.figure(figsize=[24, 6])
plt.rcParams.update({'font.size': 25})
ax = plt.subplot(1,1,1)

for key in dollar_per_block_vs_cache_type_data:
    cur_array = dollar_per_block_vs_cache_type_data[key]
    for entry in cur_array:
        dollar_per_block_list.append(entry[0])

cost_mt_st_percentage_csv = pathlib.Path("cost_mt_st_percentage_{}.csv".format(mt_type)).open("w+")
prev_edge = 0
x = []
height_array, edge_array = np.histogram(dollar_per_block_list)
for index, height in enumerate(height_array):
    cur_edge = edge_array[index:index+2]
    max_val = index*3 
    x.append(max_val-0.5)

    mt_types = ["ST-T1", "ST-T2", "MT-P", "MT-NP"]
    count_data = defaultdict(int)
    total_req = 0 
    for mt_type in mt_types:
        for entry in dollar_per_block_vs_cache_type_data[mt_type]:
            if entry[0] >= cur_edge[0] and entry[0] < cur_edge[1]:
                count_data[mt_type] += 1
                total_req += 1
    else:
        for mt_type in mt_types:
            count_data[mt_type] = 100*count_data[mt_type]/total_req 

    cost_mt_st_percentage_csv.write("{:.2f}-{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(cur_edge[0], cur_edge[1],
        count_data["ST-T1"], count_data["ST-T2"], count_data["MT-P"], count_data["MT-NP"]))
    prev_edge = cur_edge

    ax.bar([max_val-1], count_data["ST-T1"], hatch=hatch_array[0], color=(0.2, 0.4, 0.6, 0.6), label="ST-T1")
    ax.bar([max_val-1], count_data["ST-T2"], bottom=count_data["ST-T1"], hatch=hatch_array[1], color=(0.2, 0.4, 0.6, 0.6), label="ST-T2")

    ax.bar([max_val], count_data["MT-P"], hatch=hatch_array[2], color=(0.2, 0.4, 0.6, 0.6), label="MT-P")
    ax.bar([max_val], count_data["MT-NP"], bottom=count_data["MT-P"], hatch=hatch_array[3], color=(0.2, 0.4, 0.6, 0.6), label="MT-NP")

cost_mt_st_percentage_csv.close()
ax.set_xticks(x)
ax.set_xticklabels([str.format('{0:.2f}', val) for val in edge_array[1:]])

plt.ylabel("Optimal Configuration (%)")
plt.xlabel("Cost per GB working set size")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), 
    bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig('cost_hist_{}.png'.format(mt_type))
plt.close()


x = []
y = []
z = []
print(T1_LAT_DIFF)
print(T2_LAT_DIFF)
for key in mt_p_vs_np_lat_data:
    x.append(T1_LAT_DIFF[key]-T2_LAT_DIFF[key])
    y.append(T1_LAT_DIFF[key]+T2_LAT_DIFF[key])
    print(mt_p_vs_np_lat_data[key])
    z.append(np.mean(mt_p_vs_np_lat_data[key]))

sort_index = np.argsort(z)
for i in range(len(label_array)):
    print(label_array[sort_index[i]], x[sort_index[i]], y[sort_index[i]], z[sort_index[i]],
        T1_LAT_DIFF[label_array[sort_index[i]]], T2_LAT_DIFF[label_array[sort_index[i]]])




# plt.figure(figsize=[14, 10])
# plt.rcParams.update({'font.size': 25})
# ax = plt.subplot(1,1,1)
# ax.scatter(x, z)
# plt.savefig('t1_lat_diff_vs_mt_{}.png'.format(mt_type))
# plt.close()

