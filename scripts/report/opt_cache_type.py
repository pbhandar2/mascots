import argparse 
import pathlib 
import pandas as pd 


def main(data_path, write_policy):
    df = pd.read_csv(data_path)

    if not df.empty:

        df["percent_diff"] = 100*(df.loc[:, ("{}_st_t1_lat".format(write_policy))] \
            - df.loc[:, ("{}_min_lat".format(write_policy))])/df.loc[:, ("{}_st_t1_lat".format(write_policy))]

        mt_non_pyramid_df = df[(df["{}_t1".format(write_policy)]>=df["{}_t2".format(write_policy)]) & (df["{}_t2".format(write_policy)]>0)]
        mt_pyramid_df = df[(df["{}_t1".format(write_policy)]<df["{}_t2".format(write_policy)]) & (df["{}_t1".format(write_policy)]>0)]
        st_1_df = df[(df["{}_t2".format(write_policy)]<1) & (df["{}_t1".format(write_policy)]>0)]
        st_2_df = df[(df["{}_t1".format(write_policy)]<1) & (df["{}_t2".format(write_policy)]>0)]

        non_pyramid_stats = mt_non_pyramid_df["percent_diff"].describe()
        pyramid_stats = mt_pyramid_df["percent_diff"].describe()

        if non_pyramid_stats['count'] > 0:
            print(full_data_path, non_pyramid_stats['count'], non_pyramid_stats['mean'], len(df))
            # print(mt_non_pyramid_df[[
            #     "{}_t1".format(write_policy),
            #     "{}_t2".format(write_policy),
            #     "{}_st_t1".format(write_policy), 
            #     "{}_st_t1_lat".format(write_policy), 
            #     "{}_min_lat".format(write_policy)]])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The type of OPT cache for each workload")
    parser.add_argument("--data_path", 
        type=pathlib.Path, 
        default=pathlib.Path("/research2/mtc/cp_traces/mascots/cost"),
        help="Path to folder containing the data")
    parser.add_argument("--mt_config_code",
        default="D1_S1_H1",
        help="MT cache configuration")
    parser.add_argument("--write_policy",
        default="wt",
        help="Write policy")
    args = parser.parse_args()


    for i in range(1,106):
        workload_name = "w{}".format(i) if (i>=10) else "w0{}".format(i)
        full_data_path = args.data_path.joinpath(workload_name, "{}.csv".format(args.mt_config_code))
        main(full_data_path, args.write_policy)


    