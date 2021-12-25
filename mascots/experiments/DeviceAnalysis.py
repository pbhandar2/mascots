import pathlib, json 
import pandas as pd 

class DeviceAnalysis:
    """ DeviceAnalysis class allows user to generate new MT configurations 
        to evluate using device data from the internet, system measurements 
        and synthetic generation. 
    """
    def __init__(self):
        self.header_list = [
            "op",
            "size",
            "seq_percent",
            "mean_lat"
        ]


    def generate_device_fio_perf_csv(self, device_fio_perf_dir, device_fio_perf_csv):
        device_fio_perf_dir = pathlib.Path(device_fio_perf_dir)
        op_list, size_list, random_percent_list, mean_lat_list = [],[],[],[]
        for log_path in device_fio_perf_dir.iterdir():
            file_name_split = log_path.stem.split("_")
            op = file_name_split[0]
            size = int(file_name_split[1])
            random_percent = int(file_name_split[2])

            fio_json_output = json.load(log_path.open("r"))
            job = fio_json_output["jobs"][0]
            if op == "r":
                mean_lat = job["read"]["clat_ns"]["mean"]/1e3
            elif op =="w":
                mean_lat = job["write"]["clat_ns"]["mean"]/1e3

            op_list.append(op)
            size_list.append(size)
            random_percent_list.append(random_percent)
            mean_lat_list.append(mean_lat)
        
        data = {
            "op": op_list,
            "size": size_list,
            "seq_percent": random_percent_list,
            "mean_lat": mean_lat_list
        }

        df = pd.DataFrame.from_dict(data)
        df = df.sort_values(by=['op', 'size', 'seq_percent'])
        df.to_csv(device_fio_perf_csv, index=False)
