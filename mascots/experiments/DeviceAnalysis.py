import pathlib, json 

class FIOJSON:
    def __init__(self, fio_output_path):
        self.data = json.load(open(fio_output_path))

class DeviceAnalysis:
    """ DeviceAnalysis class allows user to generate new MT configurations 
        to evluate using device data from the internet, system measurements 
        and synthetic generation. 
    """
    def __init__(self):
        pass 
    
    def load_system_data(self, fio_output_dir):
        fio_output_dir = pathlib.Path(fio_output_dir)
        for output_path in fio_output_dir.iterdir():
            file_name_split = output_path.stem.split("_")
            op = file_name_split[0]
            size = int(file_name_split[1])
            random_percent = int(file_name_split[2])

            fio_json_output = json.load(output_path.open("r"))
            job = fio_json_output["jobs"][0]
            if op == "r":
                mean_lat = job["read"]["clat_ns"]["mean"]
            elif op =="w":
                mean_lat = job["write"]["clat_ns"]["mean"]

            print("{},{},{},{}".format(op, size, random_percent, mean_lat))
