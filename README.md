
# YOLOv5量化
## 说明
本项目基于 [yolov5-6.0](https://github.com/ultralytics/yolov5/tree/v6.0) 修改：
- [#1](https://github.com/Tongkaio/yolov5_quant/pull/1)：输入输出改为仅动态batch，移除多余的节点

文件说明：
- 相关函数：quantize.py
- PTQ量化脚本：ptq.py

## 运行
运行 ptq.py，导出 ptq_yolovs5.onnx： 
```shell
$ python ptq.py --eval_origin --eval_ptq

Sensitive Summary:
Top0: Using fp16 model.24, ap = 0.35677
Top1: Using fp16 model.13, ap = 0.35259
Top2: Using fp16 model.4, ap = 0.35247
Top3: Using fp16 model.8, ap = 0.35229
Top4: Using fp16 model.0, ap = 0.35226
Top5: Using fp16 model.9, ap = 0.35221
Top6: Using fp16 model.3, ap = 0.35214
Top7: Using fp16 model.2, ap = 0.35208
Top8: Using fp16 model.14, ap = 0.35200
Top9: Using fp16 model.1, ap = 0.35199
```