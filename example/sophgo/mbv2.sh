# proprecoss mbv2 final opt model
# python scripts/merge_consecutive_convs.py \
#     --input ${tpu_mlir_final_opt_onnx} \ 
#     --output ${onnx_path}

onnx_path=$1 # mbv2 onnx model path
onnx_input_name=$2 # mbv2 input name, use scripts/get_onnx_input_output.py to get it
train_img_dir=$3 # train image dir
train_bin_dir=$4 # save bin dir
train_data_number=500 # number of train data, recommend 500 or more
save_root=$5 # save root dir
calib_table=$6 # calib table path of mbv2 onnx, use tpu-mlir to generate it

# get dipoorlet data bins
python scripts/generate_data_bins.py \
    --dataset imagenet \
    --data-root ${train_img_dir} \
    --save-root ${train_bin_dir}/${onnx_input_name}

# train dipoorlet
python -m torch.distributed.launch --master_port=29500 --use_env -m dipoorlet \
        -M ${onnx_path} -I ${train_bin_dir} \
        -N ${train_data_number} -A mse -D sophgo --brecq --drop \
        -O ${save_root} --extra_calib_table ${calib_table}