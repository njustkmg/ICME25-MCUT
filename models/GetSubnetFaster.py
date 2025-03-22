import torch
import copy
import numpy as np
# 定义函数percentile，用于计算给定稀疏度下，给定分数的百分位数
def percentile(scores, sparsity):
    # 计算给定稀疏度下，给定分数的百分位数
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    # 返回给定分数的百分位数
    return scores.view(-1).kthvalue(k).values.item()

def select_max_values(scores):
    tmp = scores.view(-1).contiguous().clone()
    thres = torch.sum(tmp) * 0.2
    sorted_values, sorted_indices = torch.sort(tmp, descending=True)
    # print(sorted_values)
    # print(sorted_indices)
    r = torch.sum(torch.cumsum(sorted_values, dim = 0)<thres)
    return sorted_values[r]


class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        thres = percentile(scores, sparsity * 100)
        # thres = select_max_values(scores)
        return torch.where(scores < thres, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None