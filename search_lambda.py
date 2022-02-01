import torch
#import s3prl.hub as hub
#import soundfile as sf
import joblib
import numpy as np
import textgrid
from tqdm import tqdm
import os

# segment len penalty function
def pen(segment_length):
    return 1 - segment_length

# Simple implementation of dynamic programming based phoneme segmentation method given in
#   Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks
#   (https://arxiv.org/abs/2012.07551, INTERSPEECH 2021)
# Author: Yuan Tseng (https://github.com/roger-tseng)
def segment(reps, kmeans_model, pen, lambd=35):
    '''
    Inputs:
    reps        :   Representation sequence from self supervised model
    kmeans_model:   Pretrained scikit-learn MiniBatchKMeans model
    pen         :   penalty function penalizing segment length (longer segment, higher penalty)
    lambd       :   penalty weight (larger weight, longer segment)

    Outputs:
    boundaries  :   List of tokens at right boundaries of segments 
                    (assuming token sequence starts from 1 to Tth token)
    label_token :   List of token labels for segments

    e.g. :

    If  tokens = [34, 55, 62, 83, 42]
        boundaries = [3, 5]
        label_token = [55, 83]

    then segmentation is :
    | 34 55 62 | 83 42 |
    |    55    |   83  |

    '''
    
    # array of distances to closest cluster center, size: token sequence len * num of clusters
    distance_array = np.square( kmeans_model.transform(reps) )
    alphas = [[0, None]]

    # Perform dynamic-programming-based segmentation
    for t in range(1,reps.shape[0]+1):

        errors = []
        closest_centers = []
        
        for segment_length in range(1,t+1):

            # array len = num of clusters
            # ith element is sum of distance from the last segment_length tokens until Tth token to the ith cluster center
            distance_subarray = distance_array[t-segment_length:t].sum(axis=0)

            closest_center = distance_subarray.argmin()
            error = alphas[t-segment_length][0] + distance_subarray.min() + lambd * pen(segment_length)

            closest_centers.append(closest_center)
            errors.append(error)

        errors = np.array(errors)
        alpha, a_min, closest = errors.min(), t-1-errors.argmin(), closest_centers[errors.argmin()]
        alphas.append([alpha, a_min, closest])

    # Backtrack to find optimal boundary tokens and label
    boundaries = []
    label_tokens = []
    tk = len(alphas)-1
    while (tk!=0):
        boundaries.append(tk)
        label_tokens.append(alphas[tk][2])
        tk = alphas[tk][1]  
    boundaries.reverse()
    label_tokens.reverse()

    return boundaries, label_tokens

if __name__ == '__main__':
    manifest_path = '../spokenSQuAD/train_lambda_search_manifest.tsv'
    textgrid_dir = '../spokenSQuAD/textgrid_train'
    # load kmeans model
    kmeans_model = joblib.load('../data_transformer/models/hubert-base/hubert-base-6th-kmeans.bin')
    # read input reps
    with open(manifest_path, 'r+') as f:
        lines = f.readlines()
        lines = lines[1:]
        fileid_list = [line.strip().split('\t')[0].split('/')[1][:-4] for line in lines]
    # get phoneme seq from textgrid file
    phoneme_length_dict = {}
    for fileid in fileid_list:
        textgrid_path = os.path.join(textgrid_dir, fileid+'.TextGrid')
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phoneme_length_dict[fileid] = len(tg[1]) 
    '''
    with open('phoneme_length.tsv', 'w+') as f:
        for fileid, length in phoneme_length_dict.items():
            f.write(fileid+'\t'+str(length)+'\n')
    '''
    lambda_list = [40,100,200,300,400,500,600,700,800,900,1000]
    diff_length_list=  []
    abs_diff_length_list=  []
    # for loop
    for lambd in lambda_list:
        # init lambda value
        print(f'trying lambda value: {lambd}')
        # segmentation
        for fileid in tqdm(fileid_list):
            # load reps 
            reps_path = os.path.join('../spokenSQuAD/train_audios', fileid+'.npy')
            reps = np.load(reps_path)
            # segmentation
            boundaries, label_tokens = segment(reps, kmeans_model, pen, lambd)
            # get difference of length
            diff_length = len(label_tokens) - phoneme_length_dict[fileid]
            abs_diff_length = abs(len(label_tokens) - phoneme_length_dict[fileid])
            diff_length_list.append(diff_length)
            abs_diff_length_list.append(abs_diff_length)
        # calculate difference between length of phoneme seq and segments
        average_diff_length = sum(diff_length_list) / len(diff_length_list)
        average_abs_diff_length = sum(abs_diff_length_list) / len(abs_diff_length_list)
        print('average_diff_length:', average_diff_length)
        print('average_abs_diff_length:', average_abs_diff_length)
        if average_diff_length>0:
            print('penalty should be larger')
        elif average_diff_length<0:
            print('penalty should be smaller')
        else:
            print('optimal penalty!')
    # -------------------------------------------------
    '''
    # Read input audio
    utterance = sf.read('/home/rogert/Desktop/198_audio_clean.flac')[0]
    utterance = torch.from_numpy(utterance).to(torch.float)

    # Obtain HuBERT 6th layer representations of input
    model = getattr(hub, 'hubert')()
    model.eval()
    reps = model([utterance])['hidden_state_6'].squeeze()
    reps = reps.detach().numpy()

    # Perform k-means clustering with pretrained k-means model at https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin
    K = 100 # num of k-means clusters
    kmeans_model = joblib.load('/home/rogert/Desktop/code/km.bin')

    # Predict vector quantized tokens
    # tokens = kmeans_model.predict(reps).tolist()
    # print(tokens)

    boundaries, label_tokens = segment(reps, kmeans_model, pen, 35)
    print(boundaries)
    print(label_tokens)
    print(f"Num of segments: {len(label_tokens)}")
    '''

