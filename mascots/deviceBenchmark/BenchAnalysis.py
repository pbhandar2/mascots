import pathlib, json 
import pandas as pd 

COLUMN_LIST = ["bs", "random", "io", "mean_lat"]

class BenchAnalysis:
    def __init__(self, log_dir):
        self.log_dir = pathlib.Path(log_dir)
        self.df = pd.DataFrame(columns=COLUMN_LIST)

    def run(self):
        for data_file_path in self.log_dir.iterdir():
            print(data_file_path)

            with open(data_file_path, "r") as f:
                print(data_file_path)
                print(f.read())
                data = json.loads(f.read())

            print(data)

    