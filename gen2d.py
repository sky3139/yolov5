

import datetime
import pickle
import cv2
import numpy as np

import glob
#coding : utf-8
import os
import abc
import xml.dom.minidom as xml


# class_name = ["car", "van", "bus", "open_bed_heavy_truck", "close_bed_heavy_truck",
#               "open_bed_light_truck", "close_bed_light_truck", "others_truck", "back_plate", "side_plate"]
# # class_name = ["car", "van", "bus", "truck", "plate"]


class XmlReader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def read_content(self, filename):
        content = None
        if (False == os.path.exists(filename)):
            return content
        filehandle = None
        try:
            filehandle = open(filename, 'rb')
        except FileNotFoundError as e:
            print(e.strerror)
        try:
            content = filehandle.read()
        except IOError as e:
            print(e.strerror)
        if (None != filehandle):
            filehandle.close()
        if(None != content):
            return content.decode("utf-8", "ignore")
        return content

    @abc.abstractmethod
    def load(self, filename):
        pass


ROOT_PATH = "/home/data"  # os.getcwd()


img_s_sum = []
label_150 = []


class XmlTester(XmlReader):
    def __init__(self):
        XmlReader.__init__(self)

    def load(self, filename, use_color=False):
        filecontent = XmlReader.read_content(self, filename)
        if None != filecontent:
            # print(filename)
            dom = xml.parseString(filecontent)
            im_size = dom.getElementsByTagName('size')[0]

            im_w = int((im_size.getElementsByTagName(
                'width')[0]).childNodes[0].data)
            im_h = int((im_size.getElementsByTagName(
                "height")[0]).childNodes[0].data)
            self.im_shape = np.array([im_w, im_h])
            self.bbox = []
            self.name = dom.getElementsByTagName(
                'filename')[0].childNodes[0].data
            self.obj_name = []
            for obj in dom.getElementsByTagName('object'):
                box = obj.getElementsByTagName('bndbox')[0]

                b_xmin = float((box.getElementsByTagName(
                    "xmin")[0]).childNodes[0].data)
                b_ymin = float((box.getElementsByTagName(
                    "ymin")[0]).childNodes[0].data)
                b_xmax = float((box.getElementsByTagName(
                    "xmax")[0]).childNodes[0].data)
                b_ymax = float((box.getElementsByTagName(
                    "ymax")[0]).childNodes[0].data)
                b_name = obj.getElementsByTagName(
                    "name")[0].childNodes[0].data
                self.obj_name.append(b_name)
                # print(self.name)
                self.bbox.append([[b_xmin, b_ymin], [b_xmax, b_ymax]])

            self.polygon = []
            self.bbox = np.array(self.bbox)
            return self.im_shape, self.bbox

    def xyxy2yolo(self):
        self.bboxyolo = []
        for bx in self.bbox:  # xyxy
            cent = (bx[1] + bx[0]) / 2
            dwh = bx[1] - bx[0]
            # self.bboxyolo.append([bx[0]/self.im_shape,dwh/self.im_shape])
            self.bboxyolo.append(
                [cent / self.im_shape, dwh / self.im_shape])  # cxcywh
        return np.array(self.bboxyolo).reshape(-1, 4), self.obj_name

    def imshow(self):
        img = cv2.imread(ROOT_PATH + "/" + self.name)
        for bx in self.bbox:
            cv2.rectangle(img, bx[0], bx[1], (0, 255, 0), 1, 1)
        # cv2.imshow("a", img)
        # cv2.waitKey()
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2BGRA);
        for clas, point in self.polygon:
            # cv2.polylines(img, [point], 1, (0,0,0,255))
            # print(point)
            img = put_mask(img, point)
        size_decrease = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        img_decrease = cv2.resize(
            img, size_decrease, interpolation=cv2.INTER_CUBIC)
        # mask = im
        # cv2.imshow("Mask", mask)
        # masked = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Mask to Image", img_decrease)
        cv2.waitKey(0)


def to_color_2d():

    xmls = glob.glob("%s/1441/*.xml" % ROOT_PATH)
    for i, xml_path in enumerate(xmls):
        reader = XmlTester()
        im_shape, bbox = reader.load(xml_path, True)
    # wb是覆盖写，如果需要追加，则为‘ab'
    datas = {'img': img_s_sum, 'label': label_150}
    f = open('data.pkl', 'wb')
    data = pickle.dump(datas, f, -1)
    f.close()

    print("img len:", len(img_s_sum))


def had(fn, class_name):
    xmls = glob.glob(f"{ROOT_PATH}/{fn}/*.xml")
    for i, xml_path in enumerate(xmls):
        reader = XmlTester()
        im_shape, bbox = reader.load(xml_path)
        flb = open(xml_path[:-3] + "txt", mode="w")
        b, ns = reader.xyxy2yolo()
        for b_, target in zip(b, ns):
            flb.write("%d %f %f %f %f\n" %
                      (class_name[target], b_[0], b_[1], b_[2], b_[3]))
        flb.close()
    print(f"file:{fn} : {len(xmls)}")


def to_yolo_2d_fire():
    class_name = {'fire': 0, 'big_fire': 1, 'smoke': 2, 'others': 3}
    had("599", class_name)
    had("1225", class_name)
    had("820", class_name)


def to_yolo_2d_dianti():
    class_name = {'electric_scooter': 0, 'person': 1, 'bicycle': 2, 'others': 3}

    had("1227", class_name)
    had("1395", class_name)


start_time = datetime.datetime.now()  # 记录程序开始执行的当前时间
to_yolo_2d_fire()
stop_time = datetime.datetime.now()  # 记录执行结束的当前时间
func_time = stop_time - start_time  # 得到中间功能的运行时间
print("func is running %s s" % func_time)
