
import json
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, print_args, scale_coords)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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


def box_label(boxa, im, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    print(boxa)
    lw = 3
    # (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    p1, p2 = list(boxa)
    print(p1, p2, list(color))

    # , thickness=lw, lineType=cv2.LINE_AA)
    cv2.rectangle(im, p1, p2, [128, 125, 50], lw, cv2.LINE_AA)
    # cv2.imshow("a", im)
    # cv2.waitKey(0)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3,
                               thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, [128, 125, 50], -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

    return im


class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
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


def xyxy2lxlywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class MyModel():
    def __init__(self) -> None:
        self.opt = parse_opt()

        device = select_device('0')
        weights = "/project/train/models/weights/best.pt"
        model = DetectMultiBackend(weights, device=device)
        stride, self.names, pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(imgsz=(640, 640), s=stride)  # check image size

        # Dataloader
        model.warmup(imgsz=(1 if pt else 1, 3, *self.imgsz))  # warmup
        self.pt = pt
        self.stride = stride
        self.model = model

    @smart_inference_mode()
    def run(self,
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            vid_stride=1,  # video frame-rate stride
            ):

        seen = 0
        l = LoadImages(self.input, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
        im, im0s = l.get()
        im = torch.from_numpy(im).cuda()
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=augment, visualize=False)
    # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        s = ""
        objs = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = "./", im0s.copy()
            p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    xyxy_h = torch.tensor(xyxy).view(1, 4)
                    # normalized xywh
                    xywh = (xyxy2lxlywh(xyxy_h)).view(-1).tolist()
                    objs.append(
                        [*xywh, conf.tolist(), self.names[int(cls.tolist())]])
                    c = int(cls)  # integer clas
                    xyxy_h_pt = xyxy_h.numpy()[0].reshape(-1, 2).astype(np.int32)
                    label = f'{self.names[c]} {conf:.2f}'
                    box_label(xyxy_h_pt, im0, label)
            # cv2.imwrite("a.png", im0)
        return objs

    def __call__(self, img):
        self.input = img
        objs = self.run(**vars(self.opt))
        return objs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def init():
    model = MyModel()
    return model


def process_image(handle=None, input_image=None, args=None, ** kwargs):

    objs = handle(input_image)
    print(objs)
    obj_dict = []
    for x, y, w, h, cf, nm in objs:
        obj_dict.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "confidence": cf,
            "name": nm
        })
    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": False,
        "target_count": len(objs),
        "target_info": []
    }
    fake_result["model_data"] = {"objects": obj_dict
                                 }
    return json.dumps(fake_result, indent=4)


if __name__ == "__main__":

    mode = init()
    img = cv2.imread("/home/data/599/1014a54.jpg")
    ans = process_image(mode, img)
    print(ans)
