import torch
import quantize
import argparse
from pathlib import Path
from utils.general import print_args

FILE = Path(__file__).resolve()

def run_SensitiveAnalysis(weight, cocodir, device='cpu'):
    """敏感层分析, 打印影响较大的前10个层"""
    # prepare model
    print("Preparing Model ...")
    model = quantize.prepare_model(weight, device)
    quantize.replace_to_quantization_model(model)
    # prepare dataset
    print("Preparing Dataset ...")
    train_dataloader = quantize.prepare_dataset(cocodir, split="train")
    val_dataloader   = quantize.prepare_dataset(cocodir, split="val")
    # calibration model
    print("Calibrating ...")
    quantize.calibrate_model(model, train_dataloader, device, 200)
    # sensitive analysis
    print("Sensitive Analysis ...")
    ignored_layer = quantize.sensitive_analysis(model, val_dataloader, args.sensitive_summary)

    return ignored_layer

def run_PTQ(args, device='cpu'):
    """除敏感层(ignore_layers)之外的层进行ptq量化"""
    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weights, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers)
    # prepare dataset
    print("Prepare Dataset ....")
    train_dataloader = quantize.prepare_dataset(args.cocodir, split="train", batch_size=args.batch_size)
    val_dataloader = quantize.prepare_dataset(args.cocodir, split="val", batch_size=args.batch_size)
    # calibration model
    print("Calibrating ...")
    quantize.calibrate_model(model, train_dataloader, device, 200)
    
    summary = quantize.SummaryTool(args.ptq_summary)

    if args.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            summary.append(["Origin", ap])
    if args.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
        summary.append(["PTQ", ap])

    if args.save_ptq:
        print("Export PTQ...")
        quantize.export_ptq(model, args.ptq, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str,  default="coco", help="coco directory")
    parser.add_argument('--batch_size', type=int,  default=8, help="batch size for data loader")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--sensitive', type=bool, default=True, help="use sensitive analysis or not befor ptq")
    parser.add_argument("--sensitive_summary", type=str, default="sensitive-summary.json", help="summary save file")
    parser.add_argument("--ignore_layers", type=str, default="model\.105\.m\.(.*)", help="regx")
    
    parser.add_argument("--save_ptq", type=bool, default=True, help="file")
    parser.add_argument("--ptq", type=str, default="ptq_yolov5s.onnx", help="file")
    
    parser.add_argument("--conf_thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="iou threshold")
    
    parser.add_argument("--eval_origin", action="store_true", help="do eval for origin model")
    parser.add_argument("--eval_ptq", action="store_true", help="do eval for ptq model")
    
    parser.add_argument("--ptq_summary", type=str, default="ptq_summary.json", help="summary save file")
    
    args = parser.parse_args()
    print_args(FILE.stem, args)

    # device
    is_cuda = (args.device != "cpu") and torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    
    # 敏感层分析
    if args.sensitive:
        print("Sensitive Analysis ...")
        ignored_layer = run_SensitiveAnalysis(args.weights, args.cocodir, device)
        args.ignore_layer = list(map(lambda x: x.replace(".", r"\.") + r"\.(.*)", ignored_layer))  # 转换为正则表达式

    # PTQ并导出模型    
    print("Running PTQ ...")
    run_PTQ(args, device)
    print("PTQ Quantization done.")