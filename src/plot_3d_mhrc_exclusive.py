import argparse 
import pathlib 
import numpy as np 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

RD_HIST_PATH = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k")
RD_HIST_OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/mrc_plot/4k")
PAGE_SIZE_KB = 4
KB_IN_MB=1024


def make_3d_exclusive_mhrc(hrc):
    hrc_3d = np.zeros(shape=(len(hrc), len(hrc)))
    for rd_1 in range(len(hrc)):
        for rd_2 in range(len(hrc)):
            if rd_1 + rd_2 >= len(hrc):
                hrc_3d[rd_1][rd_2] = hrc[-1]
            else:
                hrc_3d[rd_1][rd_2] = hrc[rd_1+rd_2]
    return hrc_3d 


def get_hrc(rd_hist):
    total_req = np.sum(rd_hist)
    print("Total Req: {}".format(total_req))
    cumsum_rd_hist = np.zeros(len(rd_hist))
    cumsum_rd_hist[1:] = np.sum(np.cumsum(rd_hist[1:, :], axis=0), axis=1)
    print(cumsum_rd_hist[:5])
    return np.divide(cumsum_rd_hist, total_req)


def main(rdhist_path, bin_size):
    rd_hist = bin_rdhist(read_reuse_hist_file(rdhist_path), bin_size)
    hrc = get_hrc(rd_hist)
    mhrc = make_3d_exclusive_mhrc(hrc)
    (x, y) = np.meshgrid(np.arange(mhrc.shape[0]), np.arange(mhrc.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_wireframe(x, y, mhrc, rstride=10, cstride=10)
    plt.savefig("mhrc.png")
    print(mhrc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RD histogram")
    parser.add_argument("rd_hist_filename", help="File name of RD histogram")
    parser.add_argument("--b", default=2560, type=int, help="The bin size. Default: 2560 (equals 10MB in 4KB page)")
    args = parser.parse_args()
    main(RD_HIST_PATH.joinpath(args.rd_hist_filename), args.b)