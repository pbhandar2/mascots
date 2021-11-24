import json 
import pathlib 
import argparse 

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d

from mascots.mtCache.mtCache import MTCache
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

RD_HIST_PATH = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k")
DEVICE_LIST = []
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        DEVICE_LIST.append(json.load(f))


def make_3d_exclusive_mhrc(hrc, rd_hist, device_config):
    size = int(0.1*len(hrc))
    mt_cache = MTCache()
    hrc_3d = np.zeros(shape=(size, size))
    lat_3d = np.zeros(shape=(size, size))
    _,_,base_lat = mt_cache.eval_exclusive_mt_cache(rd_hist, 0, 0, device_config)
    
    for rd_1 in range(size):
        for rd_2 in range(size):
            if rd_1 + rd_2 >= len(hrc):
                hrc_3d[rd_1][rd_2] = hrc[-1]
            else:
                hrc_3d[rd_1][rd_2] = hrc[rd_1+rd_2]
            
            _,_,min_lat = mt_cache.eval_exclusive_mt_cache(rd_hist, rd_1, rd_2, device_config)
            lat_3d[rd_1][rd_2] = base_lat-min_lat

    return hrc_3d, lat_3d


def get_hrc(rd_hist):
    total_req = np.sum(rd_hist)
    cumsum_rd_hist = np.zeros(len(rd_hist))
    cumsum_rd_hist[1:] = np.sum(np.cumsum(rd_hist[1:, :], axis=0), axis=1)
    return np.divide(cumsum_rd_hist, total_req)


def main(rdhist_path, bin_size):
    print("3d Lat and HRC plot for {}".format(rdhist_path))
    rd_hist = bin_rdhist(read_reuse_hist_file(rdhist_path), bin_size)
    print("RDHIST loaded! length: {}".format(len(rd_hist)))


    for device_config in DEVICE_LIST:
        
        mt_label = "_".join([_["label"] for _ in device_config])
        print("Processing device {}".format(mt_label))
        
        hrc = get_hrc(rd_hist)
        mhrc, mlat = make_3d_exclusive_mhrc(hrc, rd_hist, device_config)

        rstride_val = int(len(mhrc)/40)
        cstride_val = int(len(mhrc)/40)

        (x, y) = np.meshgrid(np.arange(mhrc.shape[0]), np.arange(mhrc.shape[1]))
        
        fig=plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 28})
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_wireframe(x, y, mhrc, rstride=rstride_val, cstride=cstride_val)
        ax.set_xlabel("T1 Size (10MB)", labelpad=25)
        ax.set_ylabel("T2 Size (10MB)", labelpad=25)
        ax.set_zlabel('Hit Rate', labelpad=27)
        ax.zaxis.set_major_formatter('{x:.02f}')
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig("./plots/hrc_{}_{}.png".format(mt_label, rdhist_path.stem))
        plt.close()


        fig=plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 28})
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_wireframe(x, y, np.log(mhrc), rstride=rstride_val, cstride=cstride_val)
        ax.set_xlabel("T1 Size (10MB)", labelpad=25)
        ax.set_ylabel("T2 Size (10MB)", labelpad=25)
        ax.set_zlabel('log(Hit Rate)', labelpad=27)
        ax.zaxis.set_major_formatter('{x:.02f}')
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig("./plots/log_hrc_{}_{}.png".format(mt_label, rdhist_path.stem))
        plt.close()


        fig=plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 28})
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_wireframe(x, y, mlat, rstride=rstride_val, cstride=cstride_val)
        ax.set_xlabel("T1 Size (10MB)", labelpad=25)
        ax.set_ylabel("T2 Size (10MB)", labelpad=25)
        ax.set_zlabel('Latency Reduced (ms)', labelpad=27)
        ax.zaxis.set_major_formatter('{x:.03f}')
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig("./plots/lat_{}_{}.png".format(mt_label, rdhist_path.stem))
        plt.close()

        fig=plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 28})
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_wireframe(x, y, np.log(mlat), rstride=rstride_val, cstride=cstride_val)
        ax.set_xlabel("T1 Size (10MB)", labelpad=25)
        ax.set_ylabel("T2 Size (10MB)", labelpad=27)
        ax.set_zlabel('log(Latency Reduced (ms))', labelpad=25)
        ax.zaxis.set_major_formatter('{x:.03f}')
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig("./plots/log_lat_{}_{}.png".format(mt_label, rdhist_path.stem))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D HRC for a given RD histogram")
    parser.add_argument("workload_name", help="Name used to identify the reuse distance histogram file")
    parser.add_argument("--b", default=2560, type=int, help="The bin size. Default: 2560 (equals 10MB in 4KB page)")
    args = parser.parse_args()
    main(RD_HIST_PATH.joinpath("{}.csv".format(args.workload_name)), args.b)
