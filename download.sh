#!/bin/bash

mkdir -p llm_bmodels

cd llm_bmodels

# 下载模型
# python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/phi3-4b_int4_1dev.bmodel

# python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b_int4_seq512_1dev.bmodel

# python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int8_1dev_seq1280.bmodel

# python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpmv26_bm1684x_int4.bmodel

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm3-4b_int4_seq512_1dev.bmodel

cd ..

echo "默认模型下载完成，其他模型请根据注释自行下载。"
