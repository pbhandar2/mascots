import argparse, json, pathlib, math 
from asyncore import dispatcher_with_send 
import pandas as pd 
import numpy as np 
from tabulate import tabulate
from collections import OrderedDict, defaultdict

from get_opt_vs_algo import OPT_VS_ALGO_HEADER_LIST
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON
from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON 

with open("../../experiment_config.json") as f:
    config = json.load(f)

def get_per_workload_cost():
    cnt = 0 
    all_points_filtered = 0
    total_points_filtered = 0 
    check_count = 0 
    miss_count = 0 
    for mt_label in config["priority_2_tier_mt_label_list"]:
        error_dir = pathlib.Path(config["mhmrc_cost_analysis_dir"]).joinpath(mt_label, "wb")
        for wdir in error_dir.iterdir():
            data_path = wdir.joinpath("v2", "10", "1.csv")
            df = pd.read_csv(data_path, names=MHMRC_ROW_JSON)
            df = df.dropna()
            df = df.drop_duplicates(subset='cost', keep="first")
            all_points_filtered += len(df[df["percent_points_filtered"]==100.0])

            ex_data_path = pathlib.Path(config["ex_cost_analysis_dir"]).joinpath(mt_label, "wb", "{}.csv".format(wdir.name))
            ex_df = pd.read_csv(ex_data_path, names=OPT_ROW_JSON)
            ex_df = ex_df.dropna()

            for _, row in df[df["percent_points_filtered"]==100.0].iterrows():
                cost = row["cost"]
                ex_entry = ex_df[ex_df["cost"]==cost]
                
                if len(ex_entry)>0:
                    ex_row = ex_entry.iloc[0]
                    if ex_row["st_size"] != row["st_size"]:
                        bad_count += 1
                    assert(ex_row["st_size"] == row["st_size"])
                    min_lat = ex_row[["st_latency", "mt_p_latency", "mt_np_latency"]].min()
                    percent_diff = 100*(row["latency"]-min_lat)/min_lat
                    check_count += 1
                    print(wdir)
                    print(percent_diff)
                    print(ex_row)
                    print(row)
                    assert(percent_diff<5)
                else:
                    miss_count += 1

            total_points_filtered = df["percent_points_filtered"].sum()
            cnt += len(df)

    print("Total cnt: {}, all points removed: {}, {}".format(cnt, all_points_filtered, 100*all_points_filtered/cnt))
    print("No data found: {}, 100 filtered data found: {}".format(miss_count, check_count))
    print(all_points_filtered, 100*all_points_filtered/cnt)

get_per_workload_cost()