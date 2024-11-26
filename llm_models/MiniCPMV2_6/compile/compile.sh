#!/bin/bash
set -ex
models=""
mode=int4
quantize_args=""
name="minicpmv26"

chip="bm1684x"
num_layers=28
out_model=$name.bmodel

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --mode)
        mode="$2"
        shift 2
        ;;
    --name)
        name="$2"
        shift 2
        ;;
    *)
        echo "Invalid option: $key" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

onnx_dir=$PWD/tmp/onnx
folder='tmp/'$name'_'$chip'_'$mode
out_model=$name'_'$chip'_'$mode'.bmodel'

# Convert block
outdir=${folder}/block
mkdir -p $outdir
pushd $outdir

for ((i = 0; i < $num_layers; i++)); do
    model_transform.py \
        --model_name block_$i \
        --model_def ${onnx_dir}/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ${onnx_dir}/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel

    rm *.npz *.onnx -f

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

# convert embedding
outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ${onnx_dir}/embedding.onnx \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ${onnx_dir}/embedding.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    --model embedding_cache.bmodel

rm *.npz *.onnx -f

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd
echo $models

# convert lm_head
outdir=${folder}/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ${onnx_dir}/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --chip ${chip} \
    --model lm_head.bmodel

rm *.npz *.onnx -f

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

# Compile VIT model
outdir=${folder}/vit
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name vision_encoder \
    --model_def ${onnx_dir}/vision_transformer.onnx \
    --mlir vision_encoder.mlir

model_deploy.py \
    --mlir vision_encoder.mlir \
    --quantize BF16 \
    --processor bm1684x \
    --quant_output \
    --model vision_encoder_bf16.bmodel

rm *.npz *.onnx -f

models=${models}${outdir}'/vision_encoder_bf16.bmodel '

popd
echo $models

model_tool --combine $models -o $out_model
