import subprocess
import _pickle as pickle
import msgpack
import os
import time
import math


def log_time():
    start_time = time.time()
    return lambda x: "{3}[{0}h {1}m {2:.1f}s]".format(math.floor((time.time() - start_time)//60//60),
                                                       math.floor((time.time() - start_time)//60 -
                                                                  ((time.time() - start_time)//60//60)*60),
                                                       (time.time() - start_time) % 60, x)


def pickle_save(file, data):
    with open(file, 'wb') as output:
        pickle.dump(data, output)
        output.close()
        # subprocess.Popen(['gzip', file])
        subprocess.run(['gzip', '-f', file])


def pickle_load(file):
    # launch gunzip in the front to decompress, load pickle and return data
    subprocess.run(['gunzip', '--keep', file + '.gz'])
    with open(file, 'rb') as f:
        data = pickle.load(f)
        subprocess.run(['rm', file])
        return data


def file_exists(file):
    return os.path.exists(file + '.gz')


def msgpack_save(file, data):
    # Write msgpack file
    with open(file, 'wb') as output:
        packed = msgpack.packb(data, default=encode_this, use_bin_type=True)
        output.write(packed)
        output.close()
    # Compress msgpack with gzip keeping existing file
    with open(file + '.gz', 'wb') as output_compressed:
        subprocess.run(['gzip', '-c', '-f', file], stdout=output_compressed)
    # subprocess.run(['rm', file])


def msgpack_load(file):
    # If decompressed file doesn't exist, decompress first
    if not os.path.exists(file) and file_exists(file):
        with open(file, 'wb') as msg_file:
            subprocess.run(['gunzip', '-c', file + '.gz'], stdout=msg_file)

    with open(file, 'rb') as f:
        byte_data = f.read()
        try:
            result = msgpack.unpackb(byte_data, object_hook=decode_this, raw=False)
        except ValueError:
            result = msgpack.unpackb(byte_data, object_hook=decode_this, raw=False, strict_map_key=False)
        # subprocess.run(['rm', file])
        return result


def encode_this(obj):
    if isinstance(obj, frozenset):
        return {'__frozenset__': True, 'as_tuple': tuple(obj)}
    if isinstance(obj, set):
        return {'__set__': True, 'as_list': list(obj)}
    return obj


def decode_this(obj):
    if '__frozenset__' in obj:
        obj = frozenset(obj['as_tuple'])
    if '__set__' in obj:
        obj = set(obj['as_list'])
    return obj
