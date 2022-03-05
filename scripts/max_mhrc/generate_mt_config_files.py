import json 
import argparse 
import pathlib
from mascots.traceAnalysis.MTConfig import MTConfig 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MT cache from a list of devices")
    parser.add_argument("device_list_file", 
        help="File with list of device specifications")
    parser.add_argument("--o", 
        default=pathlib.Path("/home/pranav/mtc/mt_config"),
        type=pathlib.Path,
        help="Directory to store MT configurations")
    args = parser.parse_args()

    mt_config = MTConfig()
    mt_dict = mt_config.generate_exclusive_mt_cache(args.device_list_file)
    for num_tier in mt_dict:
        config_dir = args.o.joinpath(str(num_tier))
        config_dir.mkdir(exist_ok=True)
        mt_list = mt_dict[num_tier]
        for mt_config in mt_list:
            mt_label = "_".join([device["label"] for device in mt_config])
            mt_config_file_name = "{}.json".format(mt_label)
            mt_config_output_path = config_dir.joinpath(mt_config_file_name)
            with mt_config_output_path.open("w+") as f:
                f.write(json.dumps(mt_config, indent=4))