import torch
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, calib
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization.tensor_quant import QuantDescriptor

from utils.activations import SiLU
from utils.general import check_dataset
from utils.datasets import create_dataloader

from models.yolo import Model, Detect
from models.common import Conv
from models.experimental import attempt_load

import os
import re
import val
import json
import yaml
from tqdm import *
from pathlib import Path
from absl import logging as quant_logging


def load_yolov5_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("models/yolov5s.yaml", ch=3, nc=80, anchors=None).to(device)
    state_dict = ckpt['model'].float().state_dict()  # 有可能是fp64, 转为fp32
    model.load_state_dict(state_dict, strict=False)
    return model


# 手动initialize
# input calibrator: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")            # 创建量化描述符, 标定方式为histogram
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)     # 设置QuantConv2d的input的默认标定方式为histogram(原本默认为max)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)  # 设置QuantMaxPool2d的input的默认标定方式为histogram(原本默认为max)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)     # 设置QuantLinear的input的默认标定方式为histogram(原本默认为max)
    quant_logging.set_verbosity(quant_logging.ERROR)


def prepare_model(weight, device, auto=False):
    if auto:
        quant_modules.initialize()  # 自动插入量化节点, input和weight标定方式为max
    else:
        initialize()  # 手动initialize, 设置input的标定方式为histogram, weight的标定方式仍为max

    model = attempt_load(weight, map_location=device, inplace=True, fuse=True)

    # 简化模型
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = True
            # m.forward = m.forward_export  # assign forward (optional)

    # model = load_yolov5_model(weight, device)
    # model.float()
    # model.eval()
    # with torch.no_grad():
    #     model.fuse()  # conv bn 进行层的合并, 加速

    return model


def transfer_torch_to_quantization(nn_instance, quant_module):
    """_summary_

    Args:
        nn_instance: 实例对象, 其类型如 <class 'torch.nn.modules.conv.Conv2d'>
        quant_module: 类对象，如 <class 'pytorch_quantization.nn.modules.quant_conv.QuantConv2d'>

    Returns:
        _type_: _description_
    """
    quant_instance = quant_module.__new__(quant_module)  # 创建一个quant_module类型的实例对象
    
    for k, val in vars(nn_instance).items():  # 获取nn_instance的属性值, vars返回__dict__属性, items()返回字典的键值对, 用于迭代
        setattr(quant_instance, k, val)       # 将nn_instance的属性值赋予quant_instance

    def __init__(self):
        # 返回两个QuantDescriptor实例, self.__class__是quant_instance类，比如QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)  # 返回默认的QuantDescriptor
        if isinstance(self, quant_nn_utils.QuantInputMixin):  # 如果self是QuantInputMixin的子类, 则self只有input一个输入, 没有weight
            # 为self生成input量化器节点(TensorQuantizer==>(ONNX)QuantizeLinear和DequantizeLinear)
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                # 当pytorch-quantization>=2.1.1(with TensorRT>=8.0)可以加上这个属性进行加速，
                # 将使用 PyTorch 原生操作，并在整个计算过程中保持 x 为 PyTorch 张量。
                # 可以避免将数据转换为 NumPy 数组的开销，在 GPU 上更为高效。
                self._input_quantizer._calibrator._torch_hist = True  
        else:  # quant_nn_utils.QuantMixin, 有input和weight两个输入
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
    
    __init__(quant_instance)

    return quant_instance
    

def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False


# 递归函数
def torch_module_find_quant_module(module, module_dict, ignore_layer, prefix=''):
    for name in module._modules:
        submodule = module._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_dict, ignore_layer, prefix=path)  # 递归寻找子模块, 直到不存在子模块为止, 说明到达最小的模块
        
        submodule_id = id(type(submodule))  # 获取模块的类对象的唯一地址
        if submodule_id in module_dict:     # 如果此地址在字典里, 说明可以替换为quant module
            ignored = quantization_ignore_match(ignore_layer, path)  # 正则化匹配, 检查该模块是否需要被忽略, ignore_layer来自敏感层分析的输出
            if ignored:
                print(f"Quantization: {path} has been ignored.")
                continue
            # 转换为quant模块
            module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])


def replace_to_quantization_model(model, ignore_layer=None):
    """替换模型中的module为quant module。"""
    module_dict = {}  # dict
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:        # 遍历转换表
        # 获取类对象，例: getattr(torch.nn, 'Conv1d') ==> <class 'torch.nn.modules.conv.Conv1d'>
        module = getattr(entry.orig_mod, entry.mod_name)  
        # python中一切皆对象, id(module)将得到"类对象"<class 'torch.nn.modules.conv.Conv1d'>在内存中的地址
        # 其在程序运行中时是不变的, 例如 {94787406661216: <class 'pytorch_quantization.nn.modules.quant_conv.QuantConv1d'>, ...}
        module_dict[id(module)] = entry.replace_mod       
    
    torch_module_find_quant_module(model, module_dict, ignore_layer)


def prepare_dataset(cocodir="../datasets/coco", split="train", batch_size=4):
    if split == "train":
        path = f"{cocodir}/train2017.txt"
        augment = True
        with open("data/hyps/hyp.scratch-low.yaml") as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)
        pad = 0
    else:
        path = f"{cocodir}/val2017.txt"
        augment = False
        hyp = None
        pad = 0.5
    
    dataloader = create_dataloader(
        path=path,
        imgsz=640,
        batch_size=batch_size,
        augment=augment,  # 数据增强
        hyp=hyp,
        rect=True,
        cache=False,
        stride=32,
        pad=pad,
        image_weights=False
    )[0]
    
    return dataloader


def evaluate_coco(model, loader, save_dir="./", conf_thres=0.001, iou_thres=0.65):
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    return val.run(
        check_dataset("data/coco128.yaml"),
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        plots=False,
        save_json=False
    )[0][3]


def collect_stats(model, data_loader, device, num_batch=200):
    model.eval()  # 评估模式
    
    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:  # 如果具有校准器
                module.disable_quant()          # 禁用量化
                module.enable_calib()           # 启用校准
            else:
                module.disable()

    # test前向，搜集量化信息，前向过程中，calibrator会不断更新amax和scale
    with torch.no_grad():
        total = min(len(data_loader), num_batch)
        for i, datas in tqdm(enumerate(data_loader), total=total, desc="calibrating"):
            if (i >= total):
                break
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0   # 归一化
            model(imgs)

    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:  # 如果具有校准器
                module.enable_quant()           # 启用量化
                module.disable_calib()          # 禁用校准
            else:
                module.enable()


def compute_amax(model, device, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:  # 如果具有校准器
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)


def calibrate_model(model, dataloader, device, batch_size=200):
    """校准量化参数。"""
    # 1. 收集模型前向输出的信息
    collect_stats(model, dataloader, device, batch_size)
    # 2. 计算每个层动态范围，计算对称量化的amax, scale值
    compute_amax(model, device, method='mse')


def export_ptq(model, save_file, dynamic_batch=True):
    """导出模型。"""
    model.to('cpu')
    input_dummy = torch.randn(1, 3, 640, 640, device='cpu')
    quant_nn.TensorQuantizer.use_fb_fake_quant = True  # 导出之前需要打开fake算子
    model.eval()
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_dummy,
            save_file,
            opset_version=13,  # 至少为13才能导出量化算子
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None
        )

    quant_nn.TensorQuantizer.use_fb_fake_quant = False  # 导出之后关闭fake算子


def have_quantizer(layer):
    """判断layer是否是量化层。"""
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

    return False

class disable_quantization:
    """用于控制模型中的量化开关的上下文管理器。进入时关闭，退出时启用。"""
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled
    
    def __enter__(self):
        self.apply(disabled=True)
        
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)


class enable_quantization:
    """用于控制模型中的量化开关的上下文管理器。进入时启用，退出时关闭。"""
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
    
    def __enter__(self):
        self.apply(enabled=True)
        
    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)


class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)  # 缩进4


def sensitive_analysis(model, loader, save_file="sensitive_analysis.json"):
    """敏感层分析。

    Args:
        model : 模型
        loader (DataLoader): train_dataloader
        save_file (str, optional): json文件, 保存敏感层分析结果

    Returns:
        List: 对精度影响较大的前10个层, 形如 ["model.104", "model.37", "model.2", ..., "model.1"]
    """
    summary = SummaryTool(save_file)
    print("Sensitive analysis by each layer...")
    # 1. 遍历每个量化层，关闭该量化层并计算精度
    for i in tqdm(range(0, len(model.model))):      # for循环model的每个层
        layer = model.model[i]
        if have_quantizer(layer):                   # 如果是量化层
            with disable_quantization(layer):       # 暂时关闭该层的量化
                ap = evaluate_coco(model, loader)   # 计算精度: map
                summary.append([ap, f"model.{i}"])  # 保存精度值到json文件
                print(f"layer {i} ap: {ap}")
        else:
            print(f"ignore model.{i} since it is {type(layer)}")
    
    # 2. 保存前10个影响比较大的层的名称
    ignored_layer = []
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)  # 按ap从大到小排序
    print("Sensitive Summary:")
    for n, (ap, name) in enumerate(summary[:10]):
        ignored_layer.append(name)
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")

    return ignored_layer


if __name__ == "__main__":
    weight = "yolov5s.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader = prepare_dataset("coco", split='train')
    val_dataloader   = prepare_dataset("coco", split='val') 

    # pth_model = load_yolov5_model(weight, device)  # 原始模型
    # pth_ap = evaluate_coco(pth_model, dataloader)

    model = prepare_model(weight, device, auto=False)  # 不自动插入量化节点的模型
    # replace_to_quantization_model(model)               # 手动插入量化节点

    # 标定模型，计算标定参数
    # calibrate_model(model, train_dataloader, device, batch_size=200)
    
    # 敏感层分析
    # senstive_analysis(model, val_dataloader)
    '''
        敏感层分析步骤：
        1. for循环model的每个quantizer层
        2. 只关闭该层的量化，保留其余层的量化
        3. 验证模型精度，计算ap和map，使用evaluate_coco，并保存精度值
        4. 验证结束，重启该层的量化操作
        5. for循环所有的量化层，得到所有层的精度值
        6. 对所有层的精度值进行排序，找出最精度值影响最大的10个层，并进行打印
    '''
    # 处理敏感层分析结果：将影响较大的层关闭量化，使用fp16进行计算
    # 所以在进行ptq量化前，就要进行敏感层的分析，得到影响较大的层
    # 然后再手动插入量化节点的时候将这些影响层的量化关闭
    

    ignore_layer = ["model\.104\.(.*)", "model\.37\.(.*)", "model\.2\.(.*)", "model\.1\.(.*)", "model\.77\.(.*)",
                    "model\.99\.(.*)", "mode1\.70\.(.*)", "model\.95\.(.*)", "model\.92\.(.*)", "model\.81\.(.*)"]
    replace_to_quantization_model(model, ignore_layer) # 手动插入量化节点
    print(model)
    # export_ptq(model, "yolov5s_ptq.onnx")

    # ptq_ap = evaluate_coco(model, dataloader)