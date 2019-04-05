from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class NPZDataset(Dataset):

    def __init__(self, data_dirs, transform=None, flavour=None, limit_num_files=0):
        """
        Args: data_dirs ... a list of data directories to find files (up to 10 files read from each dir)
              transform ... a function applied to pre-process data 
              flavour ..... a string that is required to be present in the filename
              limit_num_files ... an integer limiting number of files to be taken per data directory 
        """
        self._transform = transform
        self._files = []
        
        # Load files (up to 10) from each directory in data_dirs list
        for d in data_dirs:
            file_list = [ os.path.join(d,f) for f in os.listdir(d) if flavour is None or flavour in f ]
            if limit_num_files: file_list = file_list[0:limit_num_files]
            self._files += file_list
        
        # Need to know the total number of events. Compute.
        num_events_v = [np.load(f)['labels'].shape[0] for f in self._files]
        length = np.sum(num_events_v)

        # When an event is requested, need to know which file it comes from. Create a file/event index.
        self._file_index  = np.zeros([length],dtype=np.int32)
        self._event_index = np.zeros([length],dtype=np.int32)
        ctr=0
        for findex,num_events in enumerate(num_events_v):
            self._file_index  [ctr:ctr+num_events] = findex
            self._event_index [ctr:ctr+num_events] = np.arange(num_events)
            ctr += num_events
            
    def __len__(self):
        return len(self._file_index)
    
    def __getitem__(self,idx):
        # Read data file for the specified index=idx
        f = np.load(self._files[self._file_index[idx]])
        # Retrieve event index in this file that corresponds to overall index=idx
        i = self._event_index[idx]
        # Retrieve data & label
        label = f['labels'][i]
        data  = f['event_data'][i]
        # Apply transformation function if necessary
        if self._transform is not None:
            data = self._transform(data)
        return data,label,idx


class NPZDatabuf(Dataset):

    def __init__(self, data_dirs, transform=None, flavour=None, limit_num_files=0):
        """
        Args: data_dirs ... a list of data directories to find files (up to 10 files read from each dir)
              transform ... a function applied to pre-process data 
              flavour ..... a string that is required to be present in the filename
              limit_num_files ... an integer limiting number of files to be taken per data directory 
        """
        # Create buffer
        self._data,self._label = [],[]

        # Loop over data dirs, read-in data
        kilo_ctr=0
        for d in data_dirs:
            # Create file list in this directory
            file_list = [ os.path.join(d,f) for f in os.listdir(d) if flavour is None or flavour in f ]
            if limit_num_files: file_list = file_list[0:limit_num_files]
            # Loop over files, read data
            for f in file_list:
                df = np.load(f)
                self._label += [ l for l in df['labels'] ]
                if transform is None:
                    self._data += [ d for d in df['event_data'] ]
                else:
                    self._data += [ transform(d) for d in df['event_data'] ]

                # Report progress
                if int(len(self._label)/1000.) > kilo_ctr:
                    kilo_ctr = int(len(self._label)/1000.)
                    print('Processed',kilo_ctr)

        # Report class-wise statistics
        val,ctr=np.unique(self._label,return_counts=True)
        print('Label values:',val)
        print('Statistics:',ctr)
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self,idx):
        return self._data[idx],self._label[idx],idx
    

class H5Dataset(Dataset):
    
    def __init__(self, data_dirs, transform=None, flavour=None, limit_num_files=0):
        """
        Args: data_dirs ... a list of data directories to find files (up to 10 files read from each dir)
              transform ... a function applied to pre-process data 
              flavour ..... a string that is required to be present in the filename
              limit_num_files ... an integer limiting number of files to be taken per data directory 
        """
        self._transform = transform
        self._files = []
        
        # Load files (up to 10) from each directory in data_dirs list
        for d in data_dirs:
            file_list = [ os.path.join(d,f) for f in os.listdir(d) if flavour is None or flavour in f ]
            if limit_num_files: file_list = file_list[0:limit_num_files]
            self._files += file_list

        self._file_handles = [None] * len(self._files)
        self._event_to_file_index  = []
        self._event_to_entry_index = []
        import h5py
        for file_index, file_name in enumerate(self._files):
            f = h5py.File(file_name,mode='r')
            current_size = len(self._event_to_entry_index)
            data_size    = f['event_data'].shape[0]
            self._event_to_file_index += [file_index] * data_size
            self._event_to_entry_index += range(data_size)
            f.close()
            
    def __len__(self):
        return len(self._event_to_file_index)

    def __getitem__(self,idx):
        file_index = self._event_to_file_index[idx]
        entry_index = self._event_to_entry_index[idx]
        if self._file_handles[file_index] is None:
            import h5py
            self._file_handles[file_index] = h5py.File(self._files[file_index],mode='r')
        fh = self._file_handles[file_index]
        return fh['event_data'][entry_index],fh['labels'][entry_index],idx

def Collate(batch):
    data  = np.array([sample[0] for sample in batch])
    label = np.array([sample[1] for sample in batch])
    idx   = np.array([sample[2] for sample in batch])
    return data,label,idx
