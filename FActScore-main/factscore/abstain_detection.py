import numpy as np
import re

# 表示模型无法找到信息或拒绝回答的常见短语列表
invalid_ppl_mentions = [
    "I could not find any information",
    "The search results do not provide",
    "There is no information",
    "There are no search results",
    "there are no provided search results",
    "not provided in the search results",
    "is not mentioned in the provided search results",
    "There seems to be a mistake in the question",
    "Not sources found",
    "No sources found",
    "Try a more general question"
]

def remove_citation(text):
    """
    移除文本中的引用标记（如 [1], [2]）并修正特定的短语。
    """
    # 移除类似 [1] 的引用
    text = re.sub(r"\s*\[\d+\]\s*","", text)
    # 修正特定的前缀
    if text.startswith("According to , "):
        text = text.replace("According to , ", "According to the search results, ")
    return text

def is_invalid_ppl(text):
    """
    检查文本是否以任何无效提及（表示无法回答）开始。
    """
    return np.any([text.lower().startswith(mention.lower()) for mention in invalid_ppl_mentions])

def is_invalid_paragraph_ppl(text):
    """
    检查段落是否为空，或者是否包含任何无效提及。
    """
    return len(text.strip())==0 or np.any([mention.lower() in text.lower() for mention in invalid_ppl_mentions])

def perplexity_ai_abstain_detect(generation):
    """
    针对 Perplexity AI 的生成结果进行弃权（Abstain）检测。
    如果模型表示无法找到信息，则返回 True。
    """
    output = remove_citation(generation)
    if is_invalid_ppl(output):
        return True
    
    # 检查段落，如果有段落表示无效，则认为整体弃权
    valid_paras = []
    for para in output.split("\n\n"):
        if is_invalid_paragraph_ppl(para):
            break
        valid_paras.append(para.strip())

    if len(valid_paras) == 0:
        return True
    else:
        return False

def generic_abstain_detect(generation):
    """
    通用的弃权检测，检查是否以 "I'm sorry" 开始或包含 "provide more"。
    """
    return generation.startswith("I'm sorry") or "provide more" in generation

def is_response_abstained(generation, fn_type):
    """
    根据指定的类型检测响应是否属于弃权（即模型拒绝回答或表示无相关信息）。
    :param generation: 模型生成的文本。
    :param fn_type: 检测类型 ("perplexity_ai", "generic" 或 None)。
    :return: True 表示弃权，False 表示正常响应。
    """
    if fn_type == "perplexity_ai":
        return perplexity_ai_abstain_detect(generation)

    elif fn_type == "generic":
        return generic_abstain_detect(generation)

    else:
        # 默认不进行弃权检测
        return False


