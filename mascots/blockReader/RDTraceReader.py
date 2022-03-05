from collections import Counter 

class RDTraceReader:
    def __init__(self, rd_trace_path):
        self.rd_trace_path = rd_trace_path 
    
    def generate_rd_hist_file(self, rd_hist_file):
        read_counter = Counter()
        write_counter = Counter() 
        max_read_rd = -1
        max_write_rd = -1
        with open(self.rd_trace_path) as f:
            line = f.readline().rstrip()
            while line:
                line_split = line.split(",")
                op = line_split[1]
                rd = int(line_split[0])

                if op == "r":
                    read_counter.update([rd])
                    if rd > max_read_rd:
                        max_read_rd = rd 
                else:
                    write_counter.update([rd])
                    if rd > max_write_rd:
                        max_write_rd = rd 

                line = f.readline().rstrip()

        max_rd = max(max_read_rd, max_write_rd)
        with open(rd_hist_file, "w+") as f:
            f.write("{},{}\n".format(read_counter[-1], write_counter[-1]))
            for i in range(0 ,max_rd+1):
                f.write("{},{}\n".format(read_counter[i], write_counter[i]))