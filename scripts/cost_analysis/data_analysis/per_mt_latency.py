import json, pathlib 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 
import matplotlib.pyplot as plt

cache_name_replace_map = {
    "D1": "FastDRAM",
    "D3": "SlowDRAM",
    "H3": "SlowHDD",
    "S5": "SlowSSD",
    "S3": "MediumSSD",
    "S1": "FastSSD",
    "H1": "FastHDD"
}

experiment_config="../../experiment_config.json"
with open(experiment_config, "r") as f:
    config = json.load(f)

scaler = preprocessing.MinMaxScaler()
mt_list = config["priority_2_tier_mt_label_list"]
ex_output_dir = pathlib.Path(config["ex_cost_analysis_dir"])
workload_list = list([_.stem for _ in ex_output_dir.joinpath(mt_list[0], "wb").iterdir()])
storage_device_list = list(set([_.split("_")[-1] for _ in mt_list]))

for storage_device in storage_device_list:
    for write_policy in ["wb"]:
        label_set = set()
        for workload_name in ["w98"]:
            output_path = "{}_{}_{}_per-mt-latency.csv".format(storage_device, write_policy, workload_name)

            per_cost_entry = defaultdict(lambda: defaultdict(float))
            for mt_label in mt_list:
                split_mt_label = mt_label.split("_")

                # if split_mt_label[0] != "D1":
                #     continue 

                if split_mt_label[-1] != storage_device:
                    continue 

                data_path = ex_output_dir.joinpath(mt_label, write_policy, "{}.csv".format(workload_name))
                df = pd.read_csv(data_path, names=OPT_ROW_JSON)

                for _, row in df.iterrows():
                    cost = row["cost"]
                    st_latency = row["st_latency"]
                    mt_latency = min(row["mt_p_latency"], row["mt_np_latency"])
                    
                    st_label = cache_name_replace_map[split_mt_label[0]]
                    mt_label = "{}-{}".format(cache_name_replace_map[split_mt_label[0]], cache_name_replace_map[split_mt_label[1]])

                    label_set.add(st_label)
                    label_set.add(mt_label)

                    per_cost_entry[st_label][cost] = st_latency 
                    per_cost_entry[mt_label][cost] = mt_latency 

            df = pd.DataFrame.from_dict(per_cost_entry, orient="columns")
            type_dict = {}
            for label in label_set:
                type_dict[label] = float
            df = df.astype(type_dict)
            df = df.dropna()
            df = df.sort_index()
            df['cost'] = df.index 
            df['best_mt'] = df.loc[:, df.columns != 'cost'].idxmin(axis=1)


            top_2_devices = df[label_set].apply(lambda s, n: pd.Series(s.nsmallest(n).index), axis=1, n=2)
            smallest_lat = df[top_2_devices.iloc[0, :].to_list()]

            #print(smallest_lat)

            lat_diff = []
            for _, row in smallest_lat.iterrows():
                high_value = row.max()
                low_value = row.min()
                lat_diff.append(100*(high_value - low_value)/high_value)
            df["lat_diff"] = lat_diff

            #print(df[["SlowDRAM-FastSSD", "FastDRAM-FastSSD", "SlowDRAM", "FastDRAM", "lat_diff"]][df["lat_diff"]==df["lat_diff"].max()])

            print(workload_name, storage_device)
            data = {}
            unscaled_data = {}
            devices = ["SlowDRAM-FastSSD", "FastDRAM-FastSSD", "FastDRAM", "SlowDRAM"]
            cost_list = []
            # for _, d in df.groupby(["best_mt"]):
            #     row = d[d["lat_diff"]==d["lat_diff"].max()].iloc[0]
            #     cost = row["cost"]
            #     lat_list = np.array(row[devices].to_list()).reshape(-1,1)
            #     scaled_lat_list = scaler.fit_transform(lat_list)
            #     data[cost] = [_[0] for _ in scaled_lat_list]
            #     unscaled_data[cost] = row[devices].to_list()
            #     cost_list.append(cost)


            manual_cost_array = [60, 80, 100, 120, 140]
            # manual_cost_array = list(set(np.linspace(df["cost"].min(), df["cost"].max(), 5, dtype=int)))
            print(manual_cost_array)
            print(df["cost"])
            for man_cost in manual_cost_array:
                row_100 = df[df["cost"] == man_cost][devices].iloc[0].to_list()
                unscaled_data[man_cost] = row_100

            rd1, rd2, rd3 = [],[],[] 
            cost_list += manual_cost_array
            cost_list = sorted(cost_list)

            for cost in cost_list:
                cost_data = unscaled_data[cost]
                rd1.append(cost_data[0])
                rd2.append(cost_data[1])
                rd3.append(cost_data[2])

            print(rd1, rd2, rd3)

            fig, ax = plt.subplots(figsize=(14,7))
            x = np.arange(len(unscaled_data.keys()))  # the label locations
            width = 0.2  # the width of the bars   
            rects1 = ax.bar(x - width, rd1, width, label=devices[0], hatch='++', edgecolor="black",color="palegreen")
            rects2 = ax.bar(x, rd2, width,  hatch='.0', label=devices[1], edgecolor="black",color="slateblue")
            rects3 = ax.bar(x + width, rd3, width, hatch='x', label=devices[2], edgecolor="black",color="orange")

            ax.set_xlabel("Cost ($)", fontsize=25)
            ax.set_ylabel("Mean Latency (\u03bcs)", fontsize=25)
            ax.set_xticks(range(len(unscaled_data.keys())))
            ax.set_xticklabels(cost_list)
            ax.tick_params(axis='both', which='major', labelsize=20)

            plt.tight_layout()
            plt.legend(fontsize=20)
            plt.savefig("test/{}-{}-group_hist.pdf".format(workload_name, storage_device))
            plt.close()




                


            


                    