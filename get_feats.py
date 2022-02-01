# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import numpy as np

import tqdm
import gc
import torch

import joblib
from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    get_audio_files,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    get_features, get_feature_iterator
)


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        required=True,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Features file path. You don't need to enter acoustic model details if you have dumped features",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    return parser


def main(args, logger):
    if 'train' in args.manifest_path:
        mode = 'train'
    elif 'dev' in args.manifest_path:
        mode = 'dev'
    else:
        raise ValueError('neither train nor dev are in manifest_path')
    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False

    # get filename
    _, fnames, _ = get_audio_files(args.manifest_path)

    # init dirs
    #os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)

    # Feature extraction
    if args.features_path is not None:
        logger.info(f"Loading acoustic features from {args.features_path}...")
        features_batch = np.load(args.features_path)
    else:
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        pin = 0
        # getting generator, num_files, and iterator
        generator, num_files = get_feature_iterator(
            feature_type=args.feature_type,
            checkpoint_path=args.acoustic_model_path,
            layer=args.layer,
            manifest_path=args.manifest_path,
            sample_pct=1.0,
        )
        iterator = generator()
        features_list = []
        for i, features in enumerate(tqdm.tqdm(iterator, total=num_files)): 
            features_list.append(features)
            if len(features_list)>100 or i==num_files-1:
                logger.info(
                    f"Features extracted for {len(features_list)} utterances.\n"
                )
                logger.info(
                    f"Dimensionality of representation = {features_list[0].shape[1]}"
                )


                print(f"Writing quantized predictions to {args.out_quantized_file_path}")
                for i, feats in enumerate(features_list):
                    #pred = kmeans_model.predict(feats)
                    #pred_str = " ".join(str(p) for p in pred)
                    #base_fname = os.path.basename(fnames[pin+i]).rstrip(args.extension)
                    base_fname = os.path.basename(fnames[pin+i]).split('.')[0]
                    if base_fname.endswith('_'):
                        print(fnames[pin+i])
                        print(os.path.basename(fnames[pin+i]))
                        print(base_fname)
                        print(filenameerror)
                    feat_fname = base_fname + '.npy'
                    feat_fpath = f'../spokenSQuAD/{mode}_audios/{feat_fname}'
                    np.save(feat_fpath, feats)

                gc.collect()
                torch.cuda.empty_cache()
                pin += len(features_list)
                features_list = []


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
