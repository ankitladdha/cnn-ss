#!/usr/bin/env bash
# Train a CNN

# Path for the binary of caffe. If the caffe build
# directory is in $PATH then no need to change
CAFFE_BIN=caffe.bin

FOLD=2 # Which fold of cross-validation
EXP_NAME=ari_fold_${FOLD}

# Path where all the cnn related files are there
PREFIX=/zfsauton/home/aladdha/seman_exps/ARI/data/cnn

# make the log dir
LOG_DIR=${PREFIX}/logs/${EXP_NAME}
if [ ! -d "${LOG_DIR}" ]; then
  mkdir -p ${LOG_DIR}
fi
export GLOG_log_dir=${LOG_DIR}

# Make the model dir
MODEL_DIR=${PREFIX}/models/${EXP_NAME}
if [ ! -d "${MODEL_DIR}" ]; then
  mkdir -p ${MODEL_DIR}
fi

SOLVER_CONFIG_FILE=$PREFIX/configs/solver_${EXP_NAME}.prototxt

# Initial model used to initialize a lot of layers
INIT_MODEL=$PREFIX/models/vgg16_20M.caffemodel

# Which GPU to use
DEV_ID=1
CMD="${CAFFE_BIN} train \
         --solver=${SOLVER_CONFIG_FILE} \
         --gpu=${DEV_ID} --weights=${INIT_MODEL}"
${CMD}

