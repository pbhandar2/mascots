import argparse 
import pandas as pd 
import numpy as np
import pathlib 
import json 

import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


db_label_to_plt_label = {
    "t1": "T1 Size (10MB)",
    "t2": "T2 Size (10MB)",
    "p_d": "Performance per dollar"
}


def get_hrc(rd_hist):
    total_req = np.sum(rd_hist)
    print("Total Req: {}".format(total_req))
    cumsum_rd_hist = np.zeros(len(rd_hist))
    cumsum_rd_hist[1:] = np.sum(np.cumsum(rd_hist[1:, :], axis=0), axis=1)
    return np.divide(cumsum_rd_hist, total_req)


def get_3d_exclusive_mhrc(hrc):
    hrc_3d = np.zeros(shape=(len(hrc), len(hrc)))
    for rd_1 in range(len(hrc)):
        for rd_2 in range(len(hrc)):
            if rd_1 + rd_2 >= len(hrc):
                hrc_3d[rd_1][rd_2] = hrc[-1]
            else:
                hrc_3d[rd_1][rd_2] = hrc[rd_1+rd_2]
    return hrc_3d 


def plot_3d_surface(df, x_label, y_label, z_label, output_path):
    df = df.nlargest(int(len(df)*0.01), z_label)
    fig = plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 25})
    ax = plt.subplot(1,1,1, projection='3d')

    surf = ax.plot_trisurf(df[x_label], df[y_label], np.log(df[z_label]), cmap=cm.coolwarm, norm=matplotlib.colors.LogNorm())
    fig.colorbar(surf)

    ax.set_xlabel(db_label_to_plt_label[x_label], labelpad=22)
    ax.set_ylabel(db_label_to_plt_label[y_label], labelpad=22)
    ax.set_zlabel(db_label_to_plt_label[z_label], labelpad=22)
    plt.tight_layout()
    plt.savefig(pathlib.Path("./test_analysis_plot").joinpath(output_path))
    plt.close()


def plot_scatter_3d(main_df, x_label, y_label, z_label, output_file_name):
    for sample_rate in [1,5,10,25,50,75,100]:
        df = main_df.nlargest(int(len(main_df)*(sample_rate/100)), z_label)
        fig = plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 25})
        ax = plt.subplot(1,1,1, projection='3d')

        c = np.arange(len(df))/len(df)  # create some colours
        p = ax.scatter(df[x_label], df[y_label], df[z_label], 
            alpha=0.5, c=c, cmap=plt.cm.magma)
        
        ax.set_xlabel(db_label_to_plt_label[x_label], labelpad=22)
        ax.set_ylabel(db_label_to_plt_label[y_label], labelpad=22)
        ax.set_zlabel(db_label_to_plt_label[z_label], labelpad=22)

        fig.colorbar(p, ax=ax)

        plt.tight_layout()
        plt.savefig(pathlib.Path("./test_analysis_plot/").joinpath("{}_{}".format(str(sample_rate), output_file_name)))
        plt.close()


def main(data_path):
    df = pd.read_csv(data_path, 
        names=["c", "t1", "t2", "t1_r", "t1_w", "t2_r", "t2_w", "m_r", "m_w", "mean_lat", "hit_rate", "cost", "p_d"])
    
    # remove values where there is no T1 or T2 cache 
    indexNames = df[(df["t1"]==0) & (df["t2"]==0)].index
    df.drop(indexNames , inplace=True)

    print("Plotting 3D surface ..... {}".format(data_path))
    plot_3d_surface((df, "t1", "t2", "p_d", "p_{}.png".format(data_path.stem)))
    plot_scatter_3d(df, "t1", "t2", "p_d", "s_p_{}.png".format(data_path.stem))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyze exclusive cache data")
    parser.add_argument("exclusive_cache_data_path", type=pathlib.Path, help="Path containing exclusive cache data")
    args = parser.parse_args()
    main(args.exclusive_cache_data_path)