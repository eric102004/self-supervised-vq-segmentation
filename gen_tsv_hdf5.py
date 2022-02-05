import os
import numpy as np
import h5py
from tqdm import tqdm

data_dir = '../spokenSQuAD'
lambd = 40
mode_list = ['train','dev']


for mode in mode_list:
    print(f'processing {mode} set...')
    tsv_path = os.path.join(data_dir, f'{mode}_audios_lambda{lambd}.tsv')
    hdf5_path = os.path.join(data_dir, f'{mode}_audios_lambda{lambd}.hdf5')
    subdata_dir = os.path.join(data_dir, f'{mode}_audios')
    fileid2ids_dict = {}
    for filename in tqdm(os.listdir(subdata_dir)):
        if not filename.endswith(f'.lambda{lambd}'):
            continue
        fileid = filename.split('.')[0]
        filepath = os.path.join(subdata_dir, filename)
        with open(filepath, 'r+') as f:
            ids = f.read().strip()
        fileid2ids_dict[fileid] = ids

    with open(tsv_path, 'w+') as f:
        for fileid, ids in tqdm(fileid2ids_dict.items()):
            f.write(fileid+'|'+ids+'\n')

    hf = h5py.File(hdf5_path, 'w')
    for fileid, ids in tqdm(fileid2ids_dict.items()):
        processed_ids = [int(i) for i in ids.split()]
        processed_ids = np.array(processed_ids).astype(int)
        hf.create_dataset(fileid, data=processed_ids)
    hf.close()
