import argparse 
import pathlib 
import json 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import defaultdict 


line_style_dict = {
    "D1_O1_H1": "--*",
    "D1_S1_H1": "-.o",
    "D1_S2_H1": "--v",
    "D1_O1_H2": "--D",
    "D1_S1_H2": "--+",
    "D1_S2_H2": "--1",
    "D1_S1_S2": "-->",
    "D1_O1_S2": "--o",
    "D1_O1_S1": "--2",
}


def plot_st_v_mt_lat(st_df, mt_df, output_path, device_config_name):
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 30})
    ax = plt.subplot(1,1,1)

    ax.plot(mt_df["c"], mt_df["wb_lat"], "-.^", label=device_config_name, markersize=12, alpha=0.8)
    ax.plot(st_df["c"], st_df["lat"], "-.*", label="ST", markersize=12, alpha=0.8)

    plt.xlabel("Scaled Purchase Cost ($)")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_st_v_mt_lat_wt(st_df, mt_df, output_path, device_config_name):
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 30})
    ax = plt.subplot(1,1,1)

    ax.plot(mt_df["c"], mt_df["wt_lat"], "-.^", label=device_config_name, markersize=12, alpha=0.8)
    ax.plot(st_df["c"], st_df["lat"], "-.*", label="ST", markersize=12, alpha=0.8)

    plt.xlabel("Scaled Purchase Cost ($)")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_st_v_mt_lat_per_dollar(st_df, mt_df, output_path, device_config_name):
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 30})
    ax = plt.subplot(1,1,1)

    ax.plot(mt_df["c"], mt_df["wb_d_lat_per_dollar"], "-.^", label=device_config_name, markersize=12, alpha=0.8)
    ax.plot(st_df["c"], st_df["d_lat_per_dollar"], "-.*", label="ST", markersize=12, alpha=0.8)

    plt.xlabel("Scaled Purchase Cost ($)")
    plt.ylabel("Latency Reduced per dollar (ms/$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_all_devices_lat(device_configs, data_dir):
    storage_devices = defaultdict(list)
    for device_config_path in device_configs:
        storage = device_config_path.name.split("_")
        storage_name = storage[-1]
        storage_devices[storage_name].append(device_config_path)

    for storage_device in storage_devices:
        output_path = "./plots/all_lat_{}_{}.png".format(data_dir.name, storage_device)
        list_config = storage_devices[storage_device]
        plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 30})
        ax = plt.subplot(1,1,1)
        for config in list_config:
            mt_df = pd.read_csv(config, 
                names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])
            device_config_name = config.stem 
            st_df_path = data_dir.joinpath("st_{}.csv".format(device_config_name))
            st_df = pd.read_csv(st_df_path,
                names=["c", "s", "lat", "d_lat", "d_lat_per_dollar"])

            ax.plot(mt_df["c"], mt_df["wb_d_lat"], line_style_dict[config.stem], 
                label=config.stem, markersize=12, alpha=0.8)

        ax.set_yscale("log")
        plt.xlabel("Scaled Purchase Cost ($)")
        plt.ylabel("log( (ms))")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def plot_all_devices_d_lat(device_configs, data_dir):
    storage_devices = defaultdict(list)
    for device_config_path in device_configs:
        storage = device_config_path.name.split("_")
        storage_name = storage[-1]
        storage_devices[storage_name].append(device_config_path)

    for storage_device in storage_devices:
        output_path = "./plots/all_d_lat_{}_{}.png".format(data_dir.name, storage_device)
        list_config = storage_devices[storage_device]
        plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 30})
        ax = plt.subplot(1,1,1)
        for config in list_config:
            mt_df = pd.read_csv(config, 
                names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])
            device_config_name = config.stem 
            st_df_path = data_dir.joinpath("st_{}.csv".format(device_config_name))
            st_df = pd.read_csv(st_df_path,
                names=["c", "s", "lat", "d_lat", "d_lat_per_dollar"])

            ax.plot(mt_df["c"], mt_df["wb_d_lat_per_dollar"]*1000, line_style_dict[config.stem], 
                label=config.stem, markersize=12, alpha=0.8)

        ax.set_yscale("log")
        plt.xlabel("Scaled Purchase Cost ($)")
        plt.ylabel("log(Latency Reduced per dollar (ms/$))")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main(data_dir):
    device_configs = []
    for data_file_path in data_dir.iterdir():
        if "st" not in data_file_path.name:
            device_configs.append(data_file_path)

    for device_config_path in device_configs:
        main_df = pd.read_csv(device_config_path, 
            names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])
        device_config_name = device_config_path.stem 
        st_df_path = data_dir.joinpath("st_{}.csv".format(device_config_name))
        st_df = pd.read_csv(st_df_path,
            names=["c", "s", "lat", "d_lat", "d_lat_per_dollar"])
        plot_st_v_mt_lat(st_df, main_df, "./plots/{}_{}_lat_wb.png".format(data_dir.name, device_config_name), device_config_name)
        plot_st_v_mt_lat_wt(st_df, main_df, "./plots/{}_{}_lat_wt.png".format(data_dir.name, device_config_name), device_config_name)
        plot_st_v_mt_lat_per_dollar(st_df, main_df, "./plots/{}_{}_d_lat_wb.png".format(data_dir.name, device_config_name), device_config_name)

    plot_all_devices_lat(device_configs, data_dir)
    plot_all_devices_d_lat(device_configs, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST vs MT cache of the same cost.")
    parser.add_argument("workload_name", help="The name of the workload")
    parser.add_argument("--d", default=pathlib.Path("/research2/mtc/cp_traces/exclusive_cost_data/4k"), 
        help="The directory containing MT and ST data")
    args = parser.parse_args()

    main(args.d.joinpath(args.workload_name))