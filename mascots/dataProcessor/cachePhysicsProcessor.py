import pathlib 
import logging 

from PyMimircache import Cachecow


class Error(Exception):
    """Base class for other exceptions"""
    pass


class OpCode127Error(Error):
    """Raised when the OP code in the binary trace is 127. 
    The problem is that the op code 127 can be for both 
    read and write, not clear how to handle. """
    pass


def setup_mimircache_for_raw_trace(vscsi_file_path, vscsi_type):
    """ Return a Mimircache for the correct VSCSI type. 

    Params
    ------
    vscsi_file_path: path of the input vscsi binary trace file (str)
    vscsi_type: the type of VSCSI trace (either 1 or 2)

    Return
    ------
    mimircache: a Cachecow object with the VSCSI trace loaded 
    """

    assert(vscsi_type==1 or vscsi_type==2)
    mimircache = Cachecow()
    mimircache.vscsi(vscsi_file_path, vscsi_type=vscsi_type)
    return mimircache


def vscsi_to_csv(vscsi_file_path, csv_file_path, block_size=512):
    """ Generates a CSV trace from a VSCSI binary trace file 

    Params
    ------
    vscsi_file_path: path of the input vscsi binary trace file (str)
    csv_file_path: path of the output CSV trace file (str)
    block_size: the size of an LBA (int) (optional) (default is 512)
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
            # index of size and op_code (VSCSI op_code) are different in type 1 and 2 
            """
                VSCSI Type 1: X, size, X, op code, X, LBA, time milliseconds 
                VSCSI Type 2: op_code, X, 
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
                Refer to the VSCSI manual. The problem is that the op code 127 can be for 
                both read and write. Therefore, if we find an entry with that op code in a
                trace, we require the user to handle that and make it clear weather it was a
                read or a write by manually changing it. 
            """
            if op_code == 127:
                raise OpCode127Error

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
        except OpCode127Error:
            print("OP code 127 encounted! It can be both read or write, not sure how to handle.")
    
    write_file_handle.close()
