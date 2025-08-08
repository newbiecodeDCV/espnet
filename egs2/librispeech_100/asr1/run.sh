#!/bin/bash
# run_samba_asr.sh - Training script for Samba-ASR
# Based on paper 2501.02832 implementation

set -e
set -u
set -o pipefail

# Configuration
stage=0
stop_stage=100
ngpu=1
nj=4

# Directories
dumpdir=dump
expdir=exp
datadir=data

# Training configuration
train_config=conf/train_samba_asr.yaml
decode_config=conf/decode_samba_asr.yaml

# Dataset configuration (following paper datasets)
# You can use: librispeech, gigaspeech, spgispeech
dataset="librispeech"  # Change as needed
train_set="train_clean_100"  # Adjust based on your dataset
valid_set="dev_clean"
test_sets="test_clean test_other"

# Feature type
feats_type=raw  # Raw waveform input

# Token configuration
token_type=char
nbpe=150

# Language model
use_lm=false
lm_config=conf/train_lm.yaml

echo "Samba-ASR Training Pipeline"
echo "Stage ${stage}: Data preparation"
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Preparing data directories..."

    # Create data directories if they don't exist
    for x in ${train_set} ${valid_set} ${test_sets}; do
        if [ ! -d ${datadir}/${x} ]; then
            echo "Error: ${datadir}/${x} does not exist"
            echo "Please run data preparation script first"
            exit 1
        fi
    done

    echo "Data directories verified successfully"
fi

echo "Stage ${stage}: Feature extraction and dump"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Feature extraction for training data
    for dataset_part in ${train_set} ${valid_set} ${test_sets}; do
        echo "Processing ${dataset_part}..."

        # Create dump directory
        dump_dir=${dumpdir}/${feats_type}/${dataset_part}
        mkdir -p ${dump_dir}

        # Since we're using raw waveforms, just create symbolic links
        for file in wav.scp text utt2spk spk2utt; do
            if [ -f ${datadir}/${dataset_part}/${file} ]; then
                ln -sf $(realpath ${datadir}/${dataset_part}/${file}) ${dump_dir}/
            fi
        done

        echo "Created dump for ${dataset_part}"
    done
fi

echo "Stage ${stage}: Token preparation"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Preparing token list..."

    # Create token list from training data text
    text_file=${dumpdir}/${feats_type}/${train_set}/text
    token_listdir=${dumpdir}/${feats_type}/token_list/${token_type}_nbpe${nbpe}
    mkdir -p ${token_listdir}

    # Character-level tokenization (following paper)
    if [ ${token_type} == "char" ]; then
        echo "<blank>" > ${token_listdir}/tokens.txt
        echo "<unk>" >> ${token_listdir}/tokens.txt
        cut -d' ' -f2- ${text_file} | tr ' ' '\n' | sort | uniq | grep -v "^$" >> ${token_listdir}/tokens.txt
        echo "Created character token list with $(wc -l < ${token_listdir}/tokens.txt) tokens"
    fi

    token_list=${token_listdir}/tokens.txt
fi

echo "Stage ${stage}: Language model training (optional)"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ "${use_lm}" = true ]; then
    lm_exp=${expdir}/lm_${token_type}_${nbpe}
    echo "Training language model in ${lm_exp}..."

    python3 -m espnet2.bin.lm_train \
        --config ${lm_config} \
        --train_data_path_and_name_and_type ${dumpdir}/${feats_type}/${train_set}/text,text,text \
        --valid_data_path_and_name_and_type ${dumpdir}/${feats_type}/${valid_set}/text,text,text \
        --token_list ${token_list} \
        --output_dir ${lm_exp} \
        --ngpu ${ngpu}
fi

echo "Stage ${stage}: Samba-ASR model training"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    asr_exp=${expdir}/samba_asr_${feats_type}_${token_type}_${nbpe}
    echo "Training Samba-ASR in ${asr_exp}..."

    # Set environment for custom modules
    export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

    # Training command
    python3 -m espnet2.bin.asr_train \
        --config ${train_config} \
        --train_data_path_and_name_and_type ${dumpdir}/${feats_type}/${train_set}/wav.scp,speech,sound \
        --train_data_path_and_name_and_type ${dumpdir}/${feats_type}/${train_set}/text,text,text \
        --valid_data_path_and_name_and_type ${dumpdir}/${feats_type}/${valid_set}/wav.scp,speech,sound \
        --valid_data_path_and_name_and_type ${dumpdir}/${feats_type}/${valid_set}/text,text,text \
        --token_list ${token_list} \
        --output_dir ${asr_exp} \
        --ngpu ${ngpu} \
        --num_workers ${nj} \
        --multiprocessing_distributed true \
        --dist_launcher slurm \
        --dist_init_method "file://${PWD}/.dist_init_${RANDOM}" \
        $([ -n "${CML_NODE_COUNT:-}" ] && echo "--dist_world_size ${CML_NODE_COUNT}") \
        $([ -n "${CML_NODE_RANK:-}" ] && echo "--dist_rank ${CML_NODE_RANK}")
fi

echo "Stage ${stage}: Decoding and evaluation"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    asr_exp=${expdir}/samba_asr_${feats_type}_${token_type}_${nbpe}

    # Find best model
    if [ -f ${asr_exp}/valid.acc.ave_10best.pth ]; then
        model_file=${asr_exp}/valid.acc.ave_10best.pth
    elif [ -f ${asr_exp}/valid.acc.best.pth ]; then
        model_file=${asr_exp}/valid.acc.best.pth
    else
        echo "No trained model found in ${asr_exp}"
        exit 1
    fi

    # Decode test sets
    for test_set in ${test_sets}; do
        echo "Decoding ${test_set}..."

        inference_dir=${asr_exp}/decode_${test_set}
        mkdir -p ${inference_dir}

        # Language model option
        lm_option=""
        if [ "${use_lm}" = true ]; then
            lm_exp=${expdir}/lm_${token_type}_${nbpe}
            lm_option="--lm_file ${lm_exp}/valid.loss.ave_10best.pth"
        fi

        # Inference
        python3 -m espnet2.bin.asr_inference \
            --config ${decode_config} \
            --model_file ${model_file} \
            --train_config ${asr_exp}/config.yaml \
            --data_path_and_name_and_type ${dumpdir}/${feats_type}/${test_set}/wav.scp,speech,sound \
            --key_file ${dumpdir}/${feats_type}/${test_set}/text \
            --token_list ${token_list} \
            --output_dir ${inference_dir} \
            --ngpu ${ngpu} \
            ${lm_option}

        # Calculate scores
        echo "Results for ${test_set}:"
        python3 -m espnet2.bin.tokenize_text \
            -f 2- --input ${inference_dir}/1best_recog/text \
            --output ${inference_dir}/1best_recog/text.proc \
            --token_list ${token_list}

        sclite \
            -r ${inference_dir}/1best_recog/text.proc trn \
            -h ${dumpdir}/${feats_type}/${test_set}/text.proc trn \
            -i rm -o all stdout > ${inference_dir}/result.txt

        echo "WER: $(grep -o -P 'Percent Total Error\s+=\s+\K[0-9]+\.[0-9]+' ${inference_dir}/result.txt || echo 'N/A')"
        echo "CER: $(python3 -m espnet2.bin.tokenize_text -f 2- --input ${inference_dir}/1best_recog/text --output - --token_type char | sclite -r - trn -h <(python3 -m espnet2.bin.tokenize_text -f 2- --input ${dumpdir}/${feats_type}/${test_set}/text --output - --token_type char) trn -i rm -o all stdout | grep -o -P 'Percent Total Error\s+=\s+\K[0-9]+\.[0-9]+' || echo 'N/A')"
    done
fi

echo "Stage ${stage}: Performance analysis"
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    asr_exp=${expdir}/samba_asr_${feats_type}_${token_type}_${nbpe}

    echo "=== SAMBA-ASR TRAINING RESULTS ==="
    echo "Experiment directory: ${asr_exp}"

    # Print training summary
    if [ -f ${asr_exp}/train.log ]; then
        echo "Training completed successfully"
        echo "Final training loss: $(tail -n 10 ${asr_exp}/train.log | grep -o 'loss=[0-9]*\.[0-9]*' | tail -n 1 | cut -d'=' -f2 || echo 'N/A')"
        echo "Final validation loss: $(tail -n 10 ${asr_exp}/train.log | grep -o 'validation/main/loss=[0-9]*\.[0-9]*' | tail -n 1 | cut -d'=' -f2 || echo 'N/A')"
    fi

    # Print inference results
    echo ""
    echo "=== TEST SET RESULTS ==="
    for test_set in ${test_sets}; do
        result_file=${asr_exp}/decode_${test_set}/result.txt
        if [ -f ${result_file} ]; then
            wer=$(grep -o -P 'Percent Total Error\s+=\s+\K[0-9]+\.[0-9]+' ${result_file} || echo 'N/A')
            echo "${test_set}: WER = ${wer}%"
        fi
    done

    echo ""
    echo "Model comparison with paper results:"
    echo "Paper reports - LS Clean: 1.17%, LS Other: 2.48%"
    echo "Your results above ^^^"
fi

echo "Samba-ASR training pipeline completed!"
echo "Check ${expdir}/ for trained models and results"