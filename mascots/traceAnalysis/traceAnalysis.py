

class TraceAnalysis:

    def __init__(self, trace_path):
        self.path = trace_path 
        

    def get_read_write_stats(self):
        read_page_set = set()
        write_page_set = set()
        read_count, write_count = 0, 0 
        start_time = -1
        end_time = -1
        with open(self.path) as f:
            line = f.readline().rstrip()
            while line:
                split_line = line.split(",")
                page_id, op, timestamp = int(split_line[0]), split_line[1], int(split_line[2])
                if start_time == -1:
                    start_time = timestamp
                end_time = timestamp
                if op == "r":
                    read_page_set.add(page_id)
                    read_count += 1
                elif op == "w":
                    write_page_set.add(page_id)
                    write_count += 1
                line = f.readline().rstrip()

        return {
            "workload": self.path.stem,
            "read_count": read_count, 
            "write_count": write_count,
            "total_count": read_count + write_count, 
            "read_ws": len(read_page_set),
            "write_ws": len(write_page_set),
            "total_ws": len(read_page_set)+len(write_page_set)-len(read_page_set.intersection(write_page_set)),
            "time": (end_time-start_time)/1000000
        }
