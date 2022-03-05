import json 
import itertools
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device_list = {}
    with open("device_list.json") as f:
        device_list = json.load(f)

    dram_list = [] 
    ssd_list = []
    hdd_list = []
    for device_config in device_list:
        if device_config["type"] == "DRAM":
            dram_list.append(device_config)
        elif device_config["type"] == "SSD":
            ssd_list.append(device_config)
        elif device_config["type"] == "HDD":
            hdd_list.append(device_config)
        else:
            raise ValueError("{} is not a recognized device type!".format(device_config["type"]))

    # we only consider DRAM to be tier 1 device
    tier1_device_list = dram_list 

    # we only consider SSD to be tier 2 device
    tier2_device_list = ssd_list 

    # we consider all SSD and HDD to be potential storage device 
    storage_device_list = ssd_list + hdd_list

    # generate a tuple of all tier 1 device and potential storage devices 
    cache_storage_combo = list(itertools.product(tier1_device_list, storage_device_list))

    fig, ax = plt.subplots(figsize=(14,7))
    og_ratio_list = []
    for cache_storage_config in cache_storage_combo:
        cache = cache_storage_config[0]
        storage = cache_storage_config[1]

        for ssd in ssd_list:
            # check if the tier 2 device satisfies the requirement 
            overhead = cache["read_lat"] + ssd["write_lat"]
            gain = storage["read_lat"] - ssd["read_lat"] - cache["read_lat"] \
                - ssd["write_lat"]
            og_ratio = overhead/gain 
            if gain > 0:
                mt_label = "{}_{}_{}".format(cache["label"], ssd["label"], storage["label"])
                print(mt_label)
                print("{}".format(og_ratio))
                #ax.scatter(mt_label, og_ratio)
                og_ratio_list.append(og_ratio)

    #og_ratio_list = sorted(og_ratio_list)
    #plt.plot(og_ratio_list, marker="*")
    plt.hist(og_ratio_list, 5)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("device_og_ratio.png")
    plt.close()