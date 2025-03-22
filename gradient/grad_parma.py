import numpy as np
import torch
from .mtrl.utils.types import ConfigType, TensorType
from typing import Iterable, List, Optional, Tuple
from scipy.optimize import minimize

def Integrating_gradients(grad_uni,grad_mm):
    grad = []
    grad.append(
        tuple(
            _grad.contiguous() if _grad is not None else torch.empty_like(grad_mm[k]).contiguous()   
            for k, _grad in enumerate(grad_uni)
            )
        )
    # print(grad)
    grad.append(
        tuple(
            _grad.contiguous() if _grad is not None else torch.empty_like(grad_uni[k]).contiguous()
            for k,_grad in enumerate(grad_mm)
            )
        )
    # print(grad)
    grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )  # num_tasks x dim
    # print(grad_vec.shape)
    
    return grad_vec

def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device

def apply_vector_grad_to_parameters(
    vec: TensorType, parameters: Iterable[TensorType], modal='audio_net', accumulate: bool = False
):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for name, param in parameters:
        if modal in name and 'classifier' not in name:
        # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)

            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old grad of the parameter
            if accumulate:
                param.grad = (
                    param.grad + vec[pointer : pointer + num_param].view_as(param).data
                )
            else:
                param.grad = vec[pointer : pointer + num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param
 
def cagrad_exact(cagrad_c, grad_vec, num_tasks=2):
    grads = grad_vec / 100.
    g0 = grads.mean(0)
    GG = grads.mm(grads.t())
    x_start = np.ones(num_tasks)/num_tasks
    bnds = tuple((0,1) for x in x_start)
    cons=({'type':'eq','fun':lambda x:1-sum(x)})
    A = GG.cpu().detach().numpy()
    b = x_start.copy()
    c = (cagrad_c*g0.norm()).cpu().item()
    def objfn(x):
        return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grad_vec.device)
    # print(ww)
    gw = (grads * ww.view(-1, 1)).sum(0)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-4)
    g = (g0 + lmbda * gw) / (1 + lmbda)
    return g * 100
       
def cagrad(cagrad_c, grad_vec, num_tasks=2):
    """
    grad_vec: 一个形状为 [num_tasks, dim] 的张量，代表多个任务的梯度向量。
    num_tasks: 任务的数量。
    """
    
    # 将输入的梯度向量赋值给局部变量grads
    grads = grad_vec

    # 计算梯度向量的内积矩阵，然后转移到CPU上
    GG = grads.mm(grads.t()).cpu()
    # 标准化尺度因子，用于调整内积矩阵的值
    scale = (torch.diag(GG)+1e-4).sqrt().mean()
    GG = GG / scale.pow(2)
    # 计算每个任务梯度与所有任务梯度的平均内积
    Gg = GG.mean(1, keepdims=True)
    # 计算所有梯度内积的平均值
    gg = Gg.mean(0, keepdims=True)
    
    print(GG.shape,Gg.shape, gg.shape)
    # 初始化权重w，它需要梯度
    w = torch.ones(num_tasks, 1, requires_grad=True)
    # 根据任务数量选择不同的优化器参数
    if num_tasks == 2:
        w_opt = torch.optim.SGD([w], lr=2, momentum=0.5)
    else:
        w_opt = torch.optim.SGD([w], lr=1, momentum=0.5)

    # 计算正则化常数c
    c = (gg+1e-4).sqrt() * cagrad_c

    # 初始化最优权重和最佳目标值
    w_best = None
    obj_best = np.inf
    # 优化循环
    train_epoch = 21
    for i in range(train_epoch):
        # 重置梯度
        w_opt.zero_grad()
        # 对权重应用softmax，以确保它们的和为1
        ww = torch.softmax(w, 0)
        # 计算目标函数
        obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        # 如果当前目标函数值小于最佳值，则更新最佳权重和目标值
        if obj.item() < obj_best:
            obj_best = obj.item()
            print(obj_best)
            w_best = w.clone()
            print(ww)
            
        # 如果不是最后一次迭代，则进行反向传播和优化步骤
        if i < train_epoch-1:
            obj.backward(retain_graph=True)
            w_opt.step()
            print(w_best)
            
    # 计算最优权重
    ww = torch.softmax(w_best, 0)
    # 根据最优权重计算梯度范数
    gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()
    # 计算lambda，这里可能与某种正则化或约束有关
    lmbda = c.view(-1) / (gw_norm+1e-4)
    # 计算最终的梯度，这里考虑了任务的数量和lambda
    g = ((1/num_tasks + ww * lmbda).view(-1, 1).to(grads.device) * grads).sum(0) / (1 + cagrad_c**2)
    # 返回计算出的条件梯度
    return g
