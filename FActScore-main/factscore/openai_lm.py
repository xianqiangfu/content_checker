from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):
    """
    OpenAI 语言模型类，继承自 LM 基类。
    支持调用 ChatGPT (gpt-3.5-turbo) 和 InstructGPT (text-davinci-003) 模型。
    """

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        """
        初始化 OpenAIModel。
        :param model_name: 模型名称 (如 "ChatGPT" 或 "InstructGPT")。
        :param cache_file: 缓存文件路径。
        :param key_path: 存储 OpenAI API Key 的文件路径。
        """
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        """
        从本地文件加载 API Key 并初始化模型。
        """
        key_path = self.key_path
        assert os.path.exists(key_path), f"请将您的 OpenAI API Key 放置在 {key_path} 中。"
        with open(key_path, 'r') as f:
            api_key = f.readline()
        openai.api_key = api_key.strip()
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        """
        模型生成的具体实现，根据模型类型调用不同的 API。
        """
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        
        if self.model_name == "ChatGPT":
            # 1. 构造 ChatGPT 的消息格式
            message = [{"role": "user", "content": prompt}]
            # 2. 调用 API
            response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # 3. 提取生成的文本内容
            output = response["choices"][0]["message"]["content"]
            return output, response
        elif self.model_name == "InstructGPT":
            # 1. 直接调用 Completion API
            response = call_GPT3(prompt, temp=self.temp)
            # 2. 提取生成的文本内容
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()

def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    """
    鲁棒地调用 ChatGPT API，包含自动重试逻辑。
    """
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            # 调用 ChatCompletion 接口
            response = openai.ChatCompletion.create(model=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
            received = True
        except:
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # 如果请求无效（例如 Prompt 过长），记录日志并终止
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False
            
            # 对于频率限制等错误，等待一段时间后指数级退避重试
            logging.error("API 错误: %s (%d). 等待 %d 秒" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    """
    鲁棒地调用 GPT-3 (InstructGPT) API，包含自动重试逻辑。
    """
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            # 调用 Completion 接口
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API 错误: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response

