# -*- coding: utf-8 -*-
import os
import time
import gradio as gr
from pipeline import MiniCPMV2_6
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str,  default="llm_bmodels/minicpmv26_bm1684x_int4.bmodel", help='path to the bmodel file')
parser.add_argument('-t', '--tokenizer_path', type=str, default="llm_models/MiniCPMV2_6/token_config", help='path to the tokenizer file')
parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
parser.add_argument('--port', type=int, default=8003, help='port')
args = parser.parse_args()

model = MiniCPMV2_6(args)


def gr_chat(input_str, image_str):
        history = [[input_str, None]]
        model.input_str = input_str
        # Quit
        if model.input_str in ["exit", "q", "quit"]:
            return reset()
        model.image_str = image_str

        if model.image_str:
            if not os.path.exists(model.image_str):
                word = "Can't find image: {}".format(model.image_str) 
                history[-1][1] += word
                return history
        model.encode()
        # Chat
        first_start = time.time()
        token = model.model.forward_first(
            model.input_ids, model.pixel_values, model.image_offset)
        first_end = time.time()
        tok_num = 1
        # Following tokens
        history[-1][1] = ""
        full_word_tokens = []
        while token not in [model.ID_EOS, model.ID_IM_END] and model.model.token_length < model.SEQLEN:
            full_word_tokens.append(token)
            word = model.tokenizer.decode(
                full_word_tokens, skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = model.tokenizer.decode([token, token], skip_special_tokens=True)[
                        len(pre_word):]
                history[-1][1] += word
                yield history
                full_word_tokens = []
            tok_num += 1
            token = model.model.forward_next()
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def reset():
    return [[None, None]]

def clear():
    return ""

description = """
# MiniCPMV2_6 TPU 🏁 
"""
with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="MiniCPMV2_6", height=800)

            with gr.Row():
                with gr.Column():
                    input_str = gr.Textbox(show_label=False, placeholder="Chat with MiniCPMV2_6")
                    with gr.Row():
                        submitBtn = gr.Button("Submit", variant="primary")
                        emptyBtn = gr.Button(value="Clear")
                with gr.Column():
                    image_str = gr.Image(type='filepath')
    input_str.submit(gr_chat, inputs=[input_str, image_str], outputs=chatbot).then(clear, outputs=input_str)
    submitBtn.click(gr_chat, inputs=[input_str, image_str], outputs=chatbot).then(clear, outputs=input_str)
    emptyBtn.click(reset, outputs=chatbot)

demo.queue().launch(share=False, server_name="0.0.0.0", inbrowser=True, server_port=args.port)

