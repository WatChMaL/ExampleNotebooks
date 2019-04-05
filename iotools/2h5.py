from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, h5py, tables, os

def make_file_list(data_dir,flavour=None,fstart=0,fend=1e10):
    """
    Args: data_dir ... string, path to the directory to list files
          flavour .... string, if provided, used to filter out files that do not contain this string
    """
    # Generate a file list
    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if (flavour is None or flavour in f) and f.endswith('.npz') ]
    # Order files by the suffix X in "_X.npz"
    file_map  = {}
    for f in file_list:
        file_map[int(f.rstrip('.npz').split('_')[-1])] = f
    keys = file_map.keys()
    keys.sort()
    file_list = []
    for k in keys:
        file_list.append(file_map[k])
    return file_list[fstart:fend]

def make_h5(file_list,output_file):
    """
    Args: file_list ... list, files to be taken as input
          output_file ... string, name of an output h5 file
    """

    print('Will process',len(file_list),'files...')

    # Create output file
    FILTERS   = tables.Filters(complib='zlib', complevel=5)
    output    = tables.open_file(output_file,mode='w',filters=FILTERS)
    out_ndarray = {}
    out_1darray = {}
    label     = None

    # Loop over files, read data & store
    # For labels, since it's a small 1D array, we store all at the end
    # For event_data, they will be appended file-by-file
    for file_index,file_name in enumerate(file_list):
        # Open file
        f = np.load(file_name)

        for key in f.keys():
            data_shape = f[key].shape
            if len(data_shape) < 2:
                if not key in out_1darray: out_1darray[key]=f[key].astype(np.float32)
                else: out_1darray[key] = np.hstack([out_1darray[key],f[key].astype(np.float32)])
            else:
                if not key in out_ndarray:
                    chunk_shape = [1] + list(data_shape[1:])
                    data_shape  = [0] + list(data_shape[1:])
                    out_ndarray[key] = output.create_earray(output.root,key,tables.Float32Atom(),chunkshape=chunk_shape,shape=data_shape)
                out_ndarray[key].append(f[key].astype(np.float32))

        sys.stdout.write('Progress: %1.3f\r' % (float(file_index+1)/len(file_list)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    # Create chunked-array to store 1D arrays
    for key in out_1darray:
        data = out_1darray[key]
        out_data = output.create_carray(output.root, key, tables.Float32Atom(), shape=data.shape)
        out_data[:] = data

    # Report what's stored
    print('\nFinished!\n')
    # Close output file
    output.close()

    import h5py
    f=h5py.File(output_file,mode='r')
    print('Stored keys:',f.keys())
    for key in f.keys():
        print('    %s ... shape %s' % (key,f[key].shape))

if __name__ == '__main__':
    
    data_dir = sys.argv[1]
    output_file = sys.argv[2]
    flavour = None if len(sys.argv) < 4 or sys.argv[3] == "" else sys.argv[3]
    fstart  = 0 if len(sys.argv) < 5 else int(sys.argv[4])
    fend    = 1e10 if len(sys.argv) < 6 else int(sys.argv[5])

    file_list = make_file_list(data_dir,flavour,fstart,fend)
    make_h5(file_list,output_file)




