#!/bin/bash

# 忽略警告（可选，具体实现可能依赖于环境）
# export PYTHONWARNINGS="ignore::DeprecationWarning"

# 设置默认参数
DATA_PRE_DIR="./TurNuP4/data_work/"
ENCODE_DIR="./TurNuP4/AE/"
MODEL_DIR="./TurNuP4/xgb/"
DLMODEL_DIR="./TurNuP4/dl/"
MIXMODEL_DIR="./TurNuP4/Mix_model/"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "GPU"
    DEVICE="cuda"
else
    echo "CPU！！"
    DEVICE="cpu"
fi


# 执行Python脚本
# python "${DATA_PRE_DIR}preprocessing.py"
# python "${DATA_PRE_DIR}reaction_fp.py"
# python "${DATA_PRE_DIR}seq_esm.py"
# python "${DATA_PRE_DIR}add_ESM2.py"
# python "${DATA_PRE_DIR}pos_label.py"
# python "${ENCODE_DIR}encode_2d_fl.py"

# python "${DATA_PRE_DIR}train_test_data.py" 752
# python "${DATA_PRE_DIR}similarity.py"

# 调用其他脚本
python "${DLMODEL_DIR}test.py"
python "${DLMODEL_DIR}test_del.py"
python "${MODEL_DIR}model_train.py" 1  
python "${MODEL_DIR}model_train_imb.py" 1    
python "${DLMODEL_DIR}sensitivity_analysis.py"



# chmod +x main.sh
# ./main.sh


