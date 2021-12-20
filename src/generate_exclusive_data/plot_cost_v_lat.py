import argparse 
import pathlib 
import matplotlib.pyplot as plt 
import pandas as pd 


def plot_cost_v_lat(cost_array, lat_reduced_array):
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 25})
    ax = plt.subplot(1,1,1)
    ax.plot()


def main(data_path, output_path):
    df = pd.read_csv(data_path, 
        names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])

    plot_cost_v_lat(df["c"], df["wb_dlat"])
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Cost vs Latency Reduced Curve for a workload")
    parser.add_argument("data_path", type=pathlib.Path, help="path to cost data of a workload")
    parser.add_argument("--o", type=pathlib.Path, default=pathlib.Path("./cost_v_lat"),
        help="Directory to output the plots")
    args = parser.parse_args()
    
    main(args.data_path, args.o)