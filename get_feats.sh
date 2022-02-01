export PYTHONPATH=$PYTHONPATH:`pwd`
TYPE=hubert
CKPT_PATH=../data_transformer/models/hubert-base/hubert_base_ls960.pt
KM_MODEL_PATH=../data_transformer/models/hubert-base/km.bin
LAYER=6

MANIFEST=../spokenSQuAD/train_lambda_search_manifest.tsv
OUT_QUANTIZED_FILE=../spokenSQuAD/train_audios_quantized_libri.tsv

python examples/textless_nlp/gslm/speech2unit/clustering/get_feats.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".mp3"

#MANIFEST=../spokenSQuAD/spokenSQuAD_dev_manifest.txt
#OUT_QUANTIZED_FILE=../spokenSQuAD/dev_audios_quantized_libri.tsv
#python examples/textless_nlp/gslm/speech2unit/clustering/get_feats.py \
#    --feature_type $TYPE \
#    --kmeans_model_path $KM_MODEL_PATH \
#    --acoustic_model_path $CKPT_PATH \
#    --layer $LAYER \
#    --manifest_path $MANIFEST \
#    --out_quantized_file_path $OUT_QUANTIZED_FILE \
#    --extension ".mp3"
