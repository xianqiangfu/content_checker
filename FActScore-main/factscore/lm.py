import pickle
import os
import time

class LM(object):
    """
    语言模型 (Language Model) 基类，提供通用的缓存机制。
    """

    def __init__(self, cache_file):
        """
        初始化语言模型。
        :param cache_file: 缓存文件路径（用于持久化存储模型生成结果）。
        """
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self):
        """
        加载模型到内存中（由子类实现）。
        """
        # 加载模型并赋值给 self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        """
        生成文本。首先检查缓存，如果不存在则调用模型生成。
        :param prompt: 输入提示词。
        :param sample_idx: 样本索引，用于区分同一 prompt 的多次采样。
        :param max_sequence_length: 最大输入长度。
        :param max_output_length: 最大生成长度。
        """
        prompt = prompt.strip() # 确保不以空格结尾很重要
        cache_key = f"{prompt}_{sample_idx}"

        # 1. 尝试从缓存中获取结果
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        # 2. 如果模型未加载，则加载它
        if self.model is None:
            self.load_model()

        # 3. 调用具体实现的生成逻辑
        if prompt.endswith(" True or False?\nAnswer:"):
            # 如果是判断题，只生成 1 个 token
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        # 4. 更新缓存并增加计数
        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated

    def save_cache(self):
        """
        将当前的缓存字典保存到磁盘文件中。
        """
        if self.add_n == 0:
            return

        # 重新加载最新的缓存，防止多个进程并行运行时覆盖彼此的更新
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        """
        从磁盘加载缓存文件。
        """
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle 错误: 5秒后重试...")
                    time.sleep(5)        
        else:
            # 如果文件不存在，初始化为空字典
            cache = {}
        return cache




