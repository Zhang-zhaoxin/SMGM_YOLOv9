import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo9.model import yolo_eval, yolo9_body
from yolo9.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import argparse


# 创建创建一个存储检测结果的dir
result_path = './result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# result如果之前存放的有文件，全部清除
for i in os.listdir(result_path):
    path_file = os.path.join(result_path,i)  
    if os.path.isfile(path_file):
        os.remove(path_file)

#创建一个记录检测结果的文件
txt_path =result_path + '/result.txt'
file = open(txt_path,'w')

class Yolo9(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo9_model = yolo9_body(Input(shape=(608, 608, 3)), num_anchors//3, num_classes)
        self.yolo9_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num>=2:
            self.yolo9_model = multi_gpu_model(self.yolo9_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo9_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=1):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()

        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo9_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # print(out_boxes, out_scores, out_classes)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        # all_score = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            # all_score.append(score)

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            
            # 写入检测位置 
            file.write(predicted_class + ' ' + str(score) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + ';')
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print(end - start)
        return image

if __name__ == '__main__':
    model_path = 'yolo9_weights.h5'
    anchors_path = 'model_data/yolo9_anchors.txt'
    classes_path = 'model_data/coco_classes_voc.txt'

    score = 0.5
    iou = 0.5

    model_image_size = (608, 608)
    
    parser = argparse.ArgumentParser(description='Script')
    parser.add_argument('--iteration', default=0, type=int, help='iteration')
    parser.add_argument('--attack_name', default="I-FGSM", type=str, help='attack_name')
    parser.add_argument('--attack_dataset', default="voc", type=str, help='attack_dataset')
    args = parser.parse_args()
    
    path = 'mAP/input/images-optional'  #待检测图片的位置
    # path = 'mAP/input/untargeted/{}/{}/{}'.format(args.attack_dataset, args.attack_name, args.iteration)  #待检测图片的位置

    yolo9_model = Yolo9(score, iou, anchors_path, classes_path, model_path)

    for filename in os.listdir(path):        
        image_path = path+'/'+filename
        portion = os.path.split(image_path)
        # file.write(portion[1]+' detect_result：\n') 
        file.write(portion[1]+' ')    # 这里要有一个空格        
        image = Image.open(image_path)
        r_image = yolo9_model.detect_image(image)
        file.write('\n')
        #r_image.show() 显示检测结果
        image_save_path = './result/result_'+portion[1]        
        print('detect result save to....:'+image_save_path)
        r_image.save(image_save_path)
    file.close() 
    yolo9_model.close_session()
