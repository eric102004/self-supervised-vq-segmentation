import random
import os

n_samples = 1000
data_dir = '../spokenSQuAD'
out_manifest = data_dir + '/train_lambda_search_manifest.tsv'
popular_dir = data_dir + '/train_audios'

popular_list = []

for fname in os.listdir(popular_dir):
    if fname.endswith('.mp3'):
        popular_list.append(fname)

sample_list = random.sample(popular_list, n_samples)
    
with open(out_manifest, 'w+') as f:
    f.write(data_dir+'\n')
    for sample in sample_list:
        write_name = f'train_audios/{sample}\tNone\n'
        f.write(write_name)
