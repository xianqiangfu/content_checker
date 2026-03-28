# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

class CLM(LM):
    """
    自回归语言模型 (Causal Language Model) 类，主要用于加载本地模型（如 Llama）并进行推理。
    """
    def __init__(self, model_name, model_dir, cache_file=None):
        """
        初始化 CLM。
        :param model_name: 模型名称。
        :param model_dir: 模型权重存储的本地目录。
        :param cache_file: 缓存文件路径。
        """
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        """
        加载本地模型和分词器。
        使用 8-bit 量化以减少 GPU 显存占用。
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        # 将模型转换为 int8 并移动到 GPU
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        """
        模型生成的核心实现。
        :param prompts: 输入提示词（字符串或字符串列表）。
        :param max_sequence_length: 最大序列长度。
        :param max_output_length: 最大生成长度。
        :param end_if_newline: 是否在遇到换行符时停止。
        :param end_if_second_newline: 是否在遇到第二个换行符时停止。
        :param verbose: 是否打印详细调试信息。
        :return: (生成文本, 概率得分) 或其列表。
        """
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids in input_ids:
            # 截断过长的输入
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            
            # 调用 Transformers 库进行生成
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            gen_tokens = gen_outputs["sequences"]
            
            # 提取第一个生成的 token 的 Logits (用于后续 True/False 的得分判断)
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            
            # 解码生成的文本
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            # 处理停止条件
            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores


