# Copyright 2024 Infosys Ltd.
# Use of this source code is governed by BSD-3 license that can be found in the LICENSE file or at
# https://opensource.org/license/bsd-3-clause

import argparse
import os
import re
import sys
# sys.path.append("Megatron-LM-multi-query-attention/")
import torch
from transformers import AutoModelForCausalLM
from megatron.arguments import parse_args

# Mapping of HF name w.r.t to Megatron-LM 
hf_mg_map = {
    'attn.c_proj': 'self_attention.dense',
    'mlp.c_fc': 'mlp.dense_h_to_4h',
    'mlp.c_proj': 'mlp.dense_4h_to_h',
    'attn.c_attn': 'self_attention.query_key_value',
    'attn.q_attn': 'self_attention.query',
    'attn.kv_attn': 'self_attention.key_value',
    'ln_1' : 'input_layernorm',
    'ln_2' : 'post_attention_layernorm'
}


def get_megatron_weights(hf_weights):
    embedding = {'word_embeddings' : {}, 'position_embeddings' : {}}
    encoder = {}
    embedding["word_embeddings"]["weight"] = hf_weights["transformer.wte.weight"]
    hf_weights.pop("transformer.wte.weight")
    embedding["position_embeddings"]["weight"] = hf_weights["transformer.wpe.weight"]
    hf_weights.pop("transformer.wpe.weight")
    atten_flag = 0
    temp_qw = None
    temp_kvw = None
    temp_qb = None
    temp_kvb = None
    
    layer_re = re.compile("transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    for key, value in  hf_weights.items():
        # Match the name.
        m = layer_re.match(key)
        # Stop if that's not a layer
        if m is None:
            break
        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)
        layers_prefix = f"layers.{layer_idx}"

        #Unbinding the qkv with megatron compatible
        if op_name == 'attn.c_attn':
            if atten_flag == 0:
                temp_qw, temp_kvw = value.split(2048, dim=0)
                atten_flag = 1
            else:
                temp_qb, temp_kvb = value.split(2048, dim=0)
                encoder[layers_prefix + '.self_attention.query.'+ 'weight'] = temp_qw
                encoder[layers_prefix + '.self_attention.query.'+ 'bias'] = temp_qb
                encoder[layers_prefix + '.self_attention.key_value.'+ 'weight'] = temp_kvw
                encoder[layers_prefix + '.self_attention.key_value.'+ 'bias'] = temp_kvb
                atten_flag = 0
        else:
            encoder[layers_prefix + '.' + hf_mg_map[op_name] + '.' + weight_or_bias] = value

    encoder["final_layernorm.weight"] = hf_weights["transformer.ln_f.weight"]
    encoder["final_layernorm.bias"] = hf_weights["transformer.ln_f.bias"]

    return {"embedding" : embedding, 'encoder': encoder}



def convert():
    
    args = parse_args()
    
    print("Loading HF checkpoint")
    ##Load HF checkpoint
    model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt)
    hf_weights = model.state_dict()
    
    print("Conversion started")

    megatron_weights = get_megatron_weights(hf_weights)
    
    print("Conversion completed")
    print("Saving the model")

    final_dict = {"iteration": "release", "model": {"language_model": megatron_weights},
              "checkpoint_version": 3.0, "args" : args}

    torch.save(final_dict, os.path.join(args.output_dir, "model_optim_rng.pt"))
    print("Checkpoint saved at: ", os.path.join(args.output_dir, "model_optim_rng.pt"))



if __name__ == '__main__':
    convert()