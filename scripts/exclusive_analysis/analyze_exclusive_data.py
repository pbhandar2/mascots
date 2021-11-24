import argparse 
import pathlib, json, time  
import traceback
import logging
import pandas as pd 
import numpy as np 
import multiprocessing as mp

from mascots.mtCache.mtCache import MTCache
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

START_TIME = time.time()
DEVICE_LIST = []
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        DEVICE_LIST.append(json.load(f))

DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/cost")


def process_device(df, device_config, mt_type="wb"):
    mt_label = "_".join([_["label"] for _ in device_config])
    mt_non_pyramid_df = df[(df["{}_t1".format(mt_type)]>=df["{}_t2".format(mt_type)]) & (df["{}_t2".format(mt_type)]>0)]
    mt_pyramid_df = df[(df["{}_t1".format(mt_type)]<df["{}_t2".format(mt_type)]) & (df["{}_t1".format(mt_type)]>0)]
    st_1_df = df[(df["{}_t2".format(mt_type)]<1) & (df["{}_t1".format(mt_type)]>0)]
    st_2_df = df[(df["{}_t1".format(mt_type)]<1) & (df["{}_t2".format(mt_type)]>0)]

    print(df.iloc[:1]["wb_min_lat"])

    assert(len(df) == (1+len(mt_non_pyramid_df)+len(mt_pyramid_df)+len(st_1_df)+len(st_2_df)))

    diff_non_pyramid = 100*(mt_non_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))] \
        - mt_non_pyramid_df.loc[:, ("{}_min_lat".format(mt_type))])/mt_non_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))]
    non_pyramid_stats = diff_non_pyramid.describe()

    diff_pyramid = 100*(mt_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))] \
        - mt_pyramid_df.loc[:, ("{}_min_lat".format(mt_type))])/mt_pyramid_df.loc[:, ("{}_st_t1_lat".format(mt_type))]
    pyramid_stats = diff_pyramid.describe()


    size_diff_non_pyramid = mt_non_pyramid_df.loc[:, ("{}_t1".format(mt_type))] - mt_non_pyramid_df.loc[:, ("{}_t2".format(mt_type))]
    size_diff_non_pyramid_stats = size_diff_non_pyramid.describe()

    size_diff_pyramid = mt_pyramid_df.loc[:, ("{}_t2".format(mt_type))] - mt_pyramid_df.loc[:, ("{}_t1".format(mt_type))]
    size_diff_pyramid_stats = size_diff_pyramid.describe()


    return {
        "mt_label": mt_label,
        "mt_non_pyramid_count": len(mt_non_pyramid_df),
        "mt_pyramid_count": len(mt_pyramid_df),
        "st_1_count": len(st_1_df),
        "st_2_count": len(st_2_df),
        "mean_diff_non_pyramid": non_pyramid_stats["mean"],
        "max_diff_non_pyramid": non_pyramid_stats["max"],
        "mean_diff_pyramid": pyramid_stats["mean"],
        "max_diff_pyramid": pyramid_stats["max"],
        "mean_size_diff_pyramid": size_diff_pyramid_stats["mean"],
        "mean_size_diff_non_pyramid": size_diff_non_pyramid_stats["mean"]
    }



def main(workload_name):
    entry_array = []
    for device_config in DEVICE_LIST:
        mt_label = "_".join([_["label"] for _ in device_config])
        data_path = DATA_DIR.joinpath(workload_name, "{}.csv".format(mt_label))
        df = pd.read_csv(data_path)
        entry_array.append(process_device(df, device_config))

    df = pd.DataFrame(entry_array)
    df = df.sort_values(by=["mt_non_pyramid_count"])
    print(df[["mt_label", "mt_non_pyramid_count", "mt_pyramid_count", "st_1_count", "st_2_count", "mean_diff_non_pyramid", "mean_size_diff_non_pyramid"]])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate exclusive cache analysis for a workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    args = parser.parse_args()

    main(args.workload_name)
    end = time.time()
    print("Time Elasped: {}".format((end - START_TIME)/60))