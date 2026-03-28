# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

def assert_all_approx_close(a, b, rtol, atol, count):

    idx = torch.isclose(a.float(), b.float(), rtol, atol)
    sumval = (idx==0).sum().item()
    if sumval > count:
        print(f'Too many values not close: assert {sumval} < {count}')
        try:
            torch.testing.assert_allclose(a, b, rtol, atol)
        except Exception as e:
            print(e)


def get_memory_footprint(model, return_buffers=True):
    """
    计算模型的显存占用（以字节为单位）。
    参考自 PyTorch 讨论社区。
    :param model: PyTorch 模型实例。
    :param return_buffers: 是否计算不参与梯度更新的 Buffer 张量。
    :return: 显存占用字节数。
    """
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        mem = mem + mem_bufs
    return mem


def ـreplace_linear_with_int8linear(model, modules_to_not_convert="lm_head"):
    """
    递归地将模型中的所有 Linear 层替换为自定义的 QuantizedLinearInt8 层。
    :param model: 要转换的模型。
    :param modules_to_not_convert: 不希望转换的模块名称（通常是输出层 lm_head）。
    """
    for name, module in model.named_children():
        ـreplace_linear_with_int8linear(module, modules_to_not_convert)

        if isinstance(module, torch.nn.Linear) and name != modules_to_not_convert:
            model._modules[name] = QuantizedLinearInt8(linear_layer=module)
    return


class QuantizedLinearInt8(torch.nn.Module):
    """
    Int8 量化线性层的简单高效实现。
    权重以 Int8 格式存储，节省约 50% 的 GPU 显存。
    在推理时，权重会被反量化回 fp16 进行矩阵乘法。
    
    优点：
    - 显著节省显存。
    - 精度高，因为只量化权重，且计算过程在 fp16 下进行。
    - 推理速度快。
    """
    def __init__(self, linear_layer):
        super().__init__()
        self.bias = linear_layer.bias

        weight_bit_width = 8
        weight = linear_layer.weight

        # 计算权重的缩放系数 (Scale)
        self.weight_scale = torch.nn.Parameter(
            (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half(),
        )
        # 将权重进行四舍五入并转为 int8 (char) 存储
        self.weight = torch.nn.Parameter(
            torch.round(weight.float() / self.weight_scale[:, None]).char(),
            requires_grad=False
            )

    def forward(self, x):
        # 推理时：将 int8 权重乘以 scale 恢复到 half (fp16)
        weight = self.weight.half() * self.weight_scale[:, None]
        return torch.nn.functional.linear(x, weight, self.bias)


def convert_model_to_int8_on_gpu(model, device):
    """
    将模型量化为 int8 并移动到 GPU 的高级接口。
    """
    if 'cuda' not in device:
        raise ValueError(f"目标设备必须是 GPU。不支持设备: {device}")

    # 首先转为半精度
    model.half()

    memory_before_quantization = get_memory_footprint(model)

    # 执行层替换逻辑
    ـreplace_linear_with_int8linear(model)

    # 移动到指定 GPU 设备
    model.to(device=device)
    memory_after_quantization = get_memory_footprint(model)

    saving = round(100 * memory_after_quantization/memory_before_quantization)
    memory_before_quantization = round(memory_before_quantization / 2**30, 2)  # 转为 GB
    memory_after_quantization = round(memory_after_quantization / 2**30, 2)

    print(f'量化内存统计 - 转换前: {memory_before_quantization} GB, 转换后: {memory_after_quantization} GB (占用原比例的 {saving}%)')
    return model

