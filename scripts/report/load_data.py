from numpy.core.numeric import NaN
import pandas as pd 
import pathlib 
from tabulate import tabulate
from collections import defaultdict

PAGE_SIZE = 4096
COST_ANALYSIS_DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/cost/")

mt_count_dict = defaultdict(int)
total_req_dict = defaultdict(int)
total_io_dict = defaultdict(int)

DATA = []

# loop through a workload 
for workload_dir in COST_ANALYSIS_DATA_DIR.iterdir():
    workload_name = workload_dir.stem
    total_mt_opt = 0 # number of evaluated configuration that had MT optimal cache
    total_config = 0 
    total_io = 0 # GB 
    max_mt_lat_diff = 0 # maximum difference between MT and ST cache of the same cost 
    max_lat_diff_t1 = 0 
    max_lat_diff_t2 = 0 
    max_st_t1 = 0
    max_cost = 0 
    max_lat_diff_config = NaN
    max_config_type = None

    for mt_data_file in workload_dir.iterdir():
        df = pd.read_csv(mt_data_file)
        total_config += len(df)
        # filter rows that are not MT optimal 
        mt_df = df[(df["wt_t1"]>0) & (df["wt_t2"]>0)].copy()
    
        if len(mt_df) > 0:
            sample_row = mt_df.iloc[0]
            read_count = (sample_row["wt_t1_read"] + sample_row["wt_t2_read"] + sample_row["wt_miss_read"])/1000000
            write_count = (sample_row["wt_t1_write"] + sample_row["wt_t2_write"] + sample_row["wt_miss_write"])/1000000
            req_count = read_count + write_count 

            # get total IO (read, write) in GB
            read_io_total = 1000000*(read_count * PAGE_SIZE)/(1024*1024*1024)
            write_io_total = 1000000*(write_count * PAGE_SIZE)/(1024*1024*1024)
            total_io = read_io_total + write_io_total

            # get difference between MT cache and ST cache 
            mt_df.loc[:, "lat_diff"] = 100*(mt_df["wt_st_t1_lat"]-mt_df["wt_min_lat"])/mt_df["wt_st_t1_lat"]

            if (max(mt_df["lat_diff"]) > max_mt_lat_diff):
                max_mt_lat_diff = max(mt_df["lat_diff"])
                max_mt_lat_diff_entry = mt_df[mt_df["lat_diff"]==max(mt_df["lat_diff"])]
                max_lat_diff_t1 = max_mt_lat_diff_entry["wt_t1"]
                max_lat_diff_t2 = max_mt_lat_diff_entry["wt_t2"]
                max_config_type = mt_data_file.stem
                max_st_t1 = max_mt_lat_diff_entry["wt_st_t1"]
                max_cost = max_mt_lat_diff_entry["c"]

            mt_count_dict[workload_name] += len(mt_df)
            total_mt_opt += len(mt_df)
            total_req_dict[workload_name] = req_count
            total_io_dict[workload_name] = total_io
    
    if total_mt_opt > 0:
        DATA.append([workload_name, 100*total_mt_opt/total_config, total_io, 
            100*write_io_total/total_io, req_count, max_mt_lat_diff, 
            max_lat_diff_t1, max_lat_diff_t2, max_st_t1, max_cost, max_config_type])

DATA = sorted(DATA, key=lambda x: x[2])
print(tabulate(DATA, 
    headers=[
        "Workload", 
        "MT OPT %", 
        "Total IO (GB)", 
        "Write %",
        "Total Req (Mil)", 
        "Lat Reduction %", 
        "T1",
        "T2", 
        "ST_T1",
        "COST",
        "Config"]))

# for workload_name in mt_count_dict:
#     if mt_count_dict[workload_name] > 0:
#         print("{}, {}, {:.2f}, {:.2f}".format(
#             workload_name, 
#             mt_count_dict[workload_name],
#             total_req_dict[workload_name],
#             total_io_dict[workload_name]
#         ))