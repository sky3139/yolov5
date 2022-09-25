
import ctypes
import torchvision
from contextlib import suppress
import json
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (LOGGER, check_img_size, cv2, print_args, scale_coords)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path
import time

import torch
import lltm_cpp
import numpy as np
import datetime
np.set_printoptions(suppress=True)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

model_path = "/project/train/models/v5m6_person.pt"


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)   # top left y
    y[:, 2] = w * (x[:, 2])  # bottom right x
    y[:, 3] = h * (x[:, 3])  # bottom right y
    return y


def letterbox(im, new_shape=(1280, 1280), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class LoadImages:
    def __init__(self, path, img_size=1280, stride=32, auto=True, transforms=None, vid_stride=1):
        files = []
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = 1  # number of files
        self.video_flag = [False] * ni
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.cap = None
        self.np_img = path
        self.count = 0

    def get(self):
        if self.count == self.nf:
            raise StopIteration
        # Read image
        self.count += 1
        im0 = self.np_img  # BGR
        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
        return im, im0


ctypes.LibraryLoader('./cpp/build/libmain.a')


def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates 筛选出大于阈值的数据
    x = prediction[xc]
    x[:, 5:] *= x[:, 4:5]
    conf, j = x[:, 5:].max(1, keepdim=True)
    x = x[x[:, 4].argsort(descending=True)]
    boxsrc, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
    boxes = xywh2xyxy(boxsrc)
    #     # print(boxes)
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

    return boxsrc[i], scores[i], j[i]


def xyxy2lxlywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class MyModel():
    def __init__(self) -> None:
        self.opt = parse_opt()

        weights = model_path  # "/project/train/models/weights/best.pt"
        print("load", weights)
        device = select_device('0')

        os.system(f"du -ah {weights}")
        model = DetectMultiBackend(weights, device=device)
        stride, self.names, pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(imgsz=(1280, 1280), s=stride)  # check image size

        # Dataloader
        model.warmup(imgsz=(1 if pt else 1, 3, *self.imgsz))  # warmup
        self.pt = pt
        self.stride = stride
        self.model = model

    @smart_inference_mode()
    def run(self,
            imgsz=(1280, 1280),  # inference size (height, width)
            conf_thres=0.2,  # confidence threshold
            iou_thres=0.30,  # NMS IOU threshold
            max_det=50,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            augment=False,  # augmented inference
            ):
        # seen = 0
        src_s = self.input.shape
        l = LoadImages(self.input, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=1)
        im, im0s = l.get()
        im = torch.from_numpy(im).cuda()
        im = im.float()  # uint8 to fp16/32
        im *= 1.0 / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # torch.cuda.synchronize()
        # time_start = time.perf_counter()
        pred = self.model(im, augment=augment, visualize=False)
        # torch.cuda.synchronize()
        # time_consumed = time.perf_counter() - time_start
        # print("tuili的时间: %.3f ms" % (time_consumed * 1000))
        # print(pred[0].size(),pred[1].size())
    # NMS
        # time_start = time.perf_counter()
        # torch.cuda.synchronize()
        # time_start = time.perf_counter()
        box_xywh, score, la = non_max_suppression(pred, conf_thres, iou_thres, True, max_det=max_det)
        # torch.cuda.synchronize()
        # time_consumed = time.perf_counter() - time_start
        # print("耗费的时间: %.3f ms" % (time_consumed * 1000))
        # print(box_xywh)
        xywh = xywhn2xyxy(box_xywh, src_s[1] / 640, src_s[0] / 640)
        xywh = xywh.byte().cpu().numpy()
        score = score.cpu().numpy()
        la = la.cpu().numpy()
        obj_dict = []
        for p, s, l in zip(xywh, score, la):
            _obj = {
                "x": int(p[0]),
                "y": int(p[1]),
                "width": int(p[2]),
                "height": int(p[3]),
                "confidence": float(s),
                "name": self.names[int(l)]
            }
            obj_dict.append(_obj)
        return obj_dict

    def __call__(self, img):

        self.input = img
        objs = self.run(**vars(self.opt))

        return objs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def init():
    model = MyModel()
    return model

# 模型


def process_image(handle=None, input_image=None, args=None, ** kwargs):

    obj_dict = handle(input_image)

    target_info = []
    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": len(target_info) > 0,
        "target_count": len(target_info),
        "target_info": target_info
    }
    fake_result["model_data"] = {"objects": obj_dict}

    return json.dumps(fake_result, indent=4)


if __name__ == "__main__":
    model_path = "/home/u20/yolov5/models/yolov5m6.pt"
    mode = init()
    for i in range(50):
        # img = cv2.imread("/home/data/599/1014a54.jpg")
        img = cv2.imread("/home/u20/c2/mysdk/test/zidane.jpg")
        time_start = time.perf_counter()
        ans = process_image(mode, img)
        time_consumed = time.perf_counter() - time_start
        print("总时间: %.3f ms" % (time_consumed * 1000))
        # print(ans)
        # break
