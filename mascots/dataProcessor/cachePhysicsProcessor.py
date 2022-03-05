import pathlib 
import logging 
import math 

from PyMimircache import Cachecow

class OpCode127Exception(Exception):
    """Raised when the OP code in the binary trace is 127. 
    The problem is that the op code 127 can be for both 
    read and write, not clear how to handle. """
    pass


def csv_to_page_trace(csv_file_path, page_trace_path, page_size):
    """ Generates a page trace from CSV block trace 

    Parameters
    ----------
    csv_file_path : str
        path of the input CSV trace file 
    page_trace_path : str
        path of the output page trace file 
    page_size : int 
        the size of a page in the cache in bytes
    """
    pass


def setup_mimircache_for_raw_trace(vscsi_file_path, vscsi_type):
    """ Return a Mimircache for the correct VSCSI type. 

    Parameters
    ----------
    vscsi_file_path : str
        path of the input vscsi binary trace file 
    vscsi_type : int
        the type of VSCSI trace (either 1 or 2)

    Return
    ------
    mimircache : obj
        a Cachecow object with the VSCSI trace loaded 
    """

    assert(vscsi_type==1 or vscsi_type==2)
    mimircache = Cachecow()
    mimircache.vscsi(vscsi_file_path, vscsi_type=vscsi_type)
    return mimircache


def vscsi_to_csv(vscsi_file_path, csv_file_path, block_size=512):
    """ Generates a CSV trace from a VSCSI binary trace file 

    Parameters
    ----------
    vscsi_file_path : str
        path of the input vscsi binary trace file 
    csv_file_path : str
        path of the output CSV trace file 
    block_size : int, optional
        the size of an LBA in bytes (default is 512)
    """

    logging.info("vscsi_to_csv({}, {})".format(
        vscsi_file_path, csv_file_path))

    # Read and Write Operation Codes for this trace 
    READ_OP_CODES = [40,8,168,136,9]
    WRITE_OP_CODES = [42,10,170,138,11]

    vscsi_file_path = pathlib.Path(vscsi_file_path)
    assert vscsi_file_path.is_file(), \
        "{} passed as raw trace is not a file!".format(vscsi_file_path)

    vscsi_file_name = vscsi_file_path.name
    vscsi_type = 2 if "vscsi2" in vscsi_file_name else 1 

    # setup reader and write file handle before starting to read the I/O requests
    mimircache = setup_mimircache_for_raw_trace(str(vscsi_file_path), vscsi_type)
    reader = mimircache.reader
    write_file_handle = csv_file_path.open("w+")

    io = reader.read_complete_req() 
    while io is not None:
        try:
            """
                X: unknown field 
                VSCSI Type 1: X, size, X, op code, X, LBA, time (milliseconds)
                VSCSI Type 2: op_code, X, X, size, X, LBA, time (milliseconds)
            """
            if vscsi_type == 1:
                size = int(io[1])
                op_code = int(io[3])
            else:
                size = int(io[3])
                op_code = int(io[0])
            lba = io[5]
            time_ms = int(io[6])

            """
                Refer to the VSCSI manual. The problem is that the OP code 127 
                can be for both read and write. Warn the user!  
            """
            if op_code == 127:
                raise OpCode127Exception

            if op_code in READ_OP_CODES:

                # based on VSCSI manual, if op_code is 8 and size if 0 that actually means read the next 256 blocks 
                if size == 0 and op_code == 8:
                    size = 256*block_size
                io_type = "r"
            elif op_code in WRITE_OP_CODES:

                # based on VSCSI manual, if op_code is 10 and size if 0 that actually means write the next 256 blocks 
                if size == 0 and op_code == 10:
                    size = 256*block_size
                io_type = "w"
            else:

                # ignore the op codes that do not belong to read and write 
                io = reader.read_complete_req()
                continue

            write_file_handle.write("{},{},{},{}\n".format(time_ms, lba, io_type, size))
            io = reader.read_complete_req()
        except OpCode127Exception:
            logging.warning("OP code 127 encounted! It can be both read or write, not sure how to handle. Ignoring ... ")
            io = reader.read_complete_req()
    
    write_file_handle.close()


def vscsi_to_page(vscsi_file_path, page_file_path, block_size=512, page_size=4096):
    """ Generates a CSV trace from a VSCSI binary trace file 

    Parameters
    ----------
    vscsi_file_path : str
        path of the input vscsi binary trace file 
    page_file_path : str
        path of the output page trace file 
    block_size : int, optional
        the size of an LBA in bytes (default is 512)
    """

    logging.info("vscsi_to_page({}, {})".format(
        vscsi_file_path, page_file_path))

    # Read and Write Operation Codes for this trace 
    READ_OP_CODES = [40,8,168,136,9]
    WRITE_OP_CODES = [42,10,170,138,11]

    vscsi_file_path = pathlib.Path(vscsi_file_path)
    assert vscsi_file_path.is_file(), \
        "{} passed as raw trace is not a file!".format(vscsi_file_path)

    vscsi_file_name = vscsi_file_path.name
    vscsi_type = 2 if "vscsi2" in vscsi_file_name else 1 

    # setup reader and write file handle before starting to read the I/O requests
    mimircache = setup_mimircache_for_raw_trace(str(vscsi_file_path), vscsi_type)
    reader = mimircache.reader
    write_file_handle = page_file_path.open("w+")

    io = reader.read_complete_req() 
    while io is not None:
        try:
            """
                X: unknown field 
                VSCSI Type 1: X, size, X, op code, X, LBA, time (milliseconds)
                VSCSI Type 2: op_code, X, X, size, X, LBA, time (milliseconds)
            """
            if vscsi_type == 1:
                size = int(io[1])
                op_code = int(io[3])
            else:
                size = int(io[3])
                op_code = int(io[0])
            lba = io[5]
            time_ms = int(io[6])

            """
                Refer to the VSCSI manual. The problem is that the OP code 127 
                can be for both read and write. Warn the user!  
            """
            if op_code == 127:
                raise OpCode127Exception

            if op_code in READ_OP_CODES:

                # based on VSCSI manual, if op_code is 8 and size if 0 that actually means read the next 256 blocks 
                if size == 0 and op_code == 8:
                    size = 256*block_size
                io_type = "r"
            elif op_code in WRITE_OP_CODES:

                # based on VSCSI manual, if op_code is 10 and size if 0 that actually means write the next 256 blocks 
                if size == 0 and op_code == 10:
                    size = 256*block_size
                io_type = "w"
            else:

                # ignore the op codes that do not belong to read and write 
                io = reader.read_complete_req()
                continue

            start_page = math.floor(lba*block_size/page_size)
            end_page = math.floor(((lba*block_size) + size)/page_size)

            for page_index in range(start_page, end_page+1):
                write_file_handle.write("{},{},{}\n".format(page_index, io_type, time_ms))
            io = reader.read_complete_req()
        except OpCode127Exception:
            logging.warning("OP code 127 encounted! It can be both read or write, not sure how to handle. Ignoring ... ")
            io = reader.read_complete_req()
    
    write_file_handle.close()
