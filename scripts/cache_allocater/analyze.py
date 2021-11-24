import pathlib, json 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from collections import OrderedDict

HR_DATA_PATH = pathlib.Path("./greedy_hr_data")
LAT_DATA_PATH = pathlib.Path("./greedy_lat_data")

# Load all the different device types 
DEVICE_LIST = []
MT_LABEL_LIST = []
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        device_config = json.load(f)
        DEVICE_LIST.append(device_config)
        MT_LABEL_LIST.append("_".join([_["label"] for _ in device_config]))

def main(metric="hits_per_size"):
    write_policy = "wb"
    lat_diff_percentage_array = []
    for mt_label in MT_LABEL_LIST:
        hr_file_name = "{}_{}_greedy_alloc_summary.csv".format(mt_label, metric)
        lat_file_name = "{}_{}_greedy_lat_alloc_summary.csv".format(mt_label)

        hr_df = pd.read_csv(HR_DATA_PATH.joinpath(hr_file_name), 
            names=["w", "t1", "t2", "c", "lat", "lat_p_dollar"])
        lat_df = pd.read_csv(LAT_DATA_PATH.joinpath(lat_file_name), 
            names=["w", "t1", "t2", "c", "lat", "lat_p_dollar"])

        no_cache_count = len(lat_df[(lat_df["t1"]==0) & (lat_df["t2"]==0)])
        t1_st_cache_count = len(lat_df[(lat_df["t1"]>0) & (lat_df["t2"]==0)])
        t2_st_cache_count = len(lat_df[(lat_df["t1"]==0) & (lat_df["t2"]>0)])
        p_mt_cache_count = len(lat_df[(lat_df["t1"]>0) & (lat_df["t2"]>0) & (lat_df["t1"]<=lat_df["t2"])])
        np_mt_cache_count = len(lat_df[(lat_df["t1"]>0) & (lat_df["t2"]>0) & (lat_df["t1"]>lat_df["t2"])])

        hr_lat = hr_df[hr_df["w"]=="Overall"]["lat"].item()
        lat_lat = lat_df[lat_df["w"]=="Overall"]["lat"].item()

        lat_diff = hr_lat - lat_lat 
        lat_diff_percentage = 100*lat_diff/hr_lat

        lat_diff_percentage_array.append(lat_diff_percentage)

        hr_lat_per_dollar = hr_lat/hr_df[hr_df["w"]=="Overall"]["c"].item()
        lat_lat_per_dollar = lat_lat/lat_df[lat_df["w"]=="Overall"]["c"].item()
        lat_per_dollar_diff = hr_lat_per_dollar - lat_lat_per_dollar
        lat_per_dollar_diff_percentage = 100*lat_per_dollar_diff/hr_lat_per_dollar

        print("{},{},{}".format(mt_label, lat_diff_percentage, lat_per_dollar_diff_percentage))

        print("{},{},{},{},{}\n".format(no_cache_count, t1_st_cache_count, 
            t2_st_cache_count, p_mt_cache_count, np_mt_cache_count))

    print("Mean Percentage Diff: {}".format(np.mean(lat_diff_percentage_array)))

    sort_index = np.argsort(lat_diff_percentage_array)

    print(sort_index)
    print(lat_diff_percentage_array)

    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 28})
    ax = plt.subplot(1,1,1)

    for i, index in enumerate(sort_index):
        ax.bar(i, lat_diff_percentage_array[index], color="blue")

    ax.set_xticks(range(len(sort_index)))
    ax.set_xticklabels([MT_LABEL_LIST[i] for i in sort_index], rotation=90)
    ax.set_ylabel("Latency Reduction (%)")
    ax.set_xlabel("MT Configuration")
    plt.tight_layout()
    plt.savefig("hr_vs_lat_hist.png")
    plt.close()



def main2(metric="hits_per_size"):

    write_policy = "wb"
    lat_diff_percentage_array = []
    hr_lat_array = []
    lat_lat_array = []

    hr_lat_per_dollar_array = []
    lat_lat_per_dollar_array = []

    for mt_label in MT_LABEL_LIST:
        hr_file_name = "{}_{}_greedy_alloc_summary.csv".format(mt_label, metric)
        lat_file_name = "{}_greedy_lat_alloc_summary.csv".format(mt_label)

        hr_df = pd.read_csv(HR_DATA_PATH.joinpath(hr_file_name), 
            names=["w", "t1", "t2", "c", "lat", "lat_p_dollar"])
        lat_df = pd.read_csv(LAT_DATA_PATH.joinpath(lat_file_name), 
            names=["w", "t1", "t2", "c", "lat", "lat_p_dollar"])

        no_cache_count = len(lat_df[(lat_df["t1"]==0) & (lat_df["t2"]==0)])
        t1_st_cache_count = len(lat_df[(lat_df["t1"]>0) & (lat_df["t2"]==0)])
        t2_st_cache_count = len(lat_df[(lat_df["t1"]==0) & (lat_df["t2"]>0)])
        p_mt_cache_count = len(lat_df[(lat_df["t1"]>0) & (lat_df["t2"]>0) & (lat_df["t1"]<=lat_df["t2"])])
        np_mt_cache_count = len(lat_df[(lat_df["t1"]>0) & (lat_df["t2"]>0) & (lat_df["t1"]>lat_df["t2"])])

        hr_no_cache_count = len(hr_df[(lat_df["t1"]==0) & (hr_df["t2"]==0)])
        hr_t1_st_cache_count = len(hr_df[(lat_df["t1"]>0) & (hr_df["t2"]==0)])
        hr_t2_st_cache_count = len(hr_df[(lat_df["t1"]==0) & (hr_df["t2"]>0)])
        hr_p_mt_cache_count = len(hr_df[(lat_df["t1"]>0) & (hr_df["t2"]>0) & (hr_df["t1"]<=hr_df["t2"])])
        hr_np_mt_cache_count = len(hr_df[(lat_df["t1"]>0) & (hr_df["t2"]>0) & (hr_df["t1"]>hr_df["t2"])])


        hr_lat = hr_df[hr_df["w"]=="Overall"]["lat"].item()
        lat_lat = lat_df[lat_df["w"]=="Overall"]["lat"].item()

        hr_lat_array.append(hr_lat)
        lat_lat_array.append(lat_lat)

        lat_diff = hr_lat - lat_lat 
        lat_diff_percentage = 100*lat_diff/hr_lat

        lat_diff_percentage_array.append(lat_diff_percentage)

        hr_lat_per_dollar = hr_lat/hr_df[hr_df["w"]=="Overall"]["c"].item()
        lat_lat_per_dollar = lat_lat/lat_df[lat_df["w"]=="Overall"]["c"].item()\

        hr_lat_per_dollar_array.append(hr_lat_per_dollar)
        lat_lat_per_dollar_array.append(lat_lat_per_dollar)

        lat_per_dollar_diff = hr_lat_per_dollar - lat_lat_per_dollar
        lat_per_dollar_diff_percentage = 100*lat_per_dollar_diff/hr_lat_per_dollar

        

        print("{},{},{}".format(mt_label, lat_diff_percentage, lat_per_dollar_diff_percentage))

        print("{},{},{},{},{}\n".format(no_cache_count, t1_st_cache_count, 
            t2_st_cache_count, p_mt_cache_count, np_mt_cache_count))
        print("{},{},{},{},{}\n".format(hr_no_cache_count, hr_t1_st_cache_count, 
            hr_t2_st_cache_count, hr_p_mt_cache_count, hr_np_mt_cache_count))

    #print("Mean Percentage Diff: {}".format(np.mean(lat_diff_percentage_array)))

    sort_index = np.argsort(lat_diff_percentage_array)

    #print(sort_index)
    #print(lat_diff_percentage_array)
    print([lat_diff_percentage_array[_] for _ in sort_index])

    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 28})
    ax = plt.subplot(1,1,1)

    for i, index in enumerate(sort_index):
        ax.bar(i, hr_lat_array[index], color="blue", width=0.25, label="HRC Allocation", 
        hatch="//", align='center')
        ax.bar(i+0.25, lat_lat_array[index], color="red", width=0.25, label="Latency-Cost Allocation", 
        hatch="*", align='center')

    ax.set_xticks([_+0.125 for _ in range(len(sort_index))])
    ax.set_xticklabels([MT_LABEL_LIST[i] for i in sort_index], rotation=90)
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_xlabel("MT Configuration")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), 
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=4, mode="expand", borderaxespad=0.)

    plt.tight_layout()
    plt.savefig("hr_vs_lat_hist_2.png")
    plt.close()


if __name__ == "__main__":
    main2()