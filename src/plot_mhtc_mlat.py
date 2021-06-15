import argparse 
import pathlib 
import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def f(df, x, y):
    return df[df["t1_size"]==x & df["t1_size"]==y]


def main(data_file):
    df = pd.read_csv(data_file, names=["config", "t1_size", "t2_size", "t1_read", "t1_write", "t2_read", "t2_write",
        "miss_read", "miss_write", "mean_lat", "hrc"])
    
    print(df[:5])

    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 25})
    ax = plt.subplot(1,1,1, projection='3d')

    filter_df = df[(df["t1_size"]>0) | (df["t2_size"]>0)]
    print(filter_df[:3])

    pnt3d = ax.scatter3D(
        filter_df['t1_size'],
        filter_df['t2_size'], 
        filter_df['mean_lat'],
        c=filter_df['mean_lat'])
    cbar=plt.colorbar(pnt3d)

    print(filter_df[filter_df["mean_lat"]==filter_df['mean_lat'].max()])

    # (x, y) = np.meshgrid(filter_df["t1_size"], filter_df["t2_size"])

    # print(x)
    # print(y)
    # print(filter_df[["t1_size", "mean_lat"]].to_numpy())

    # surf = ax.plot_surface(x, y, filter_df[["t1_size", "mean_lat"]].to_numpy(), cmap='viridis', edgecolor='none')

    plt.savefig("3d_mean_lat.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3d HRC and Latency surface plot")
    parser.add_argument("data_file", help="Path to data file")
    args = parser.parse_args()

    main(args.data_file)