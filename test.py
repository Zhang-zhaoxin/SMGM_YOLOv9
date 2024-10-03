import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from yolo9.model import yolo_eval, yolo9_body
from yolo9.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
graph = tf.get_default_graph()

def letterbox_image_1(image, w, h, nw, nh):
    """
    resize image with unchanged aspect ratio using padding
    图像截取
    """
    iw, ih = image.size

    # if iw > ih:
    #     cbox = [(w-nw)//2, (h-nh)//2, w, (h-nh)//2 + nh]
    # else:
    cbox = [(w-nw)//2, (h-nh)//2, (w-nw)//2 + nw, (h-nh)//2 + nh]

    image_cropped = image.crop(cbox)

    return image_cropped

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
        self.yolo9_model = yolo9_body(Input(shape=(640, 640, 3)), num_anchors // 3, num_classes)
        self.yolo9_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num >= 2:
            self.yolo9_model = multi_gpu_model(self.yolo9_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2,))
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

    def detect_image(self, image, model_image_size=(640, 640)):
        start = timer()

        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo9_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print(out_boxes, out_scores, out_classes)
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
            if 0 <= c <= 5:
                with open('train.txt', 'a', encoding='utf-8') as f:
                    f.write(str("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}".format(left, top, right, bottom, c)) + ' ')
            if c==7:
                c_1 = c - 1
                with open('train.txt', 'a', encoding='utf-8') as f:
                    f.write(str("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}".format(left, top, right, bottom, c_1)) + ' ')
                    
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

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
        # b = len(all_score)
        # total = 0
        # for ele in range(0, b):
        # total = total + all_score[ele]
        # average = total / b

        end = timer()
        # print(end - start)
        return image, out_classes

    def Attack(self, image, attack_name, dataset, count, jpgfile, model_image_size=(640, 640)):
        # sess = K.get_session()
        global graph
        # start = timer()
        ori_image = image # 扰动截取之用
        pixdata_1 = ori_image.load()
        
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        original_image = np.copy(image_data)
        # with graph.as_default():
        object_hack = 0  # 该参数为要攻击的目标类别
        A = self.classes >= object_hack
        B = self.classes <= object_hack
        hack_scores = tf.boolean_mask(self.scores, A & B)
        cost_function = tf.reduce_sum(self.scores)  # 全部目标攻击为隐身
        # print("cost_function:{}".format(cost_function))
        gradient_function = K.gradients(cost_function, self.yolo9_model.input)[0]
        cost = 1
        e = 4 / 255
        n = 0
        index = 0
        time_sum = 0
        top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
        top_list= []
        left_list= []
        bottom_list= []
        right_list= []
        # 最大改变幅度
        max_change_above = original_image + 0.06
        max_change_below = original_image - 0.06
        # 初始化梯度
        pre_g = np.zeros(image_data.shape)
        gradients_m = np.zeros(image_data.shape)
        # 主要攻击循环
        # while cost > 0.002:
        for i in range(0, 10):
            img = image_data[0]
            img *= 255.
            im = Image.fromarray(img.astype(np.uint8))
            im = letterbox_image_1(im, w, h, nw, nh)
            im = im.resize((iw, ih), Image.BICUBIC)
            '''裁剪扰动'''
            # im = im.convert("RGB")
            pixdata = im.load()
            for i_width in range(iw):#遍历图片的所有像素
                for j_height in range(ih):
                    if i_width < left_min or i_width > right_max or j_height < top_min or j_height > bottom_max:
                        pixdata[i_width,j_height] = pixdata_1[i_width,j_height]

            '''临时用'''
            im.save(os.path.join("output", os.path.basename(jpgfile)))
            # 再次打开图片
            im = Image.open(os.path.join("output", os.path.basename(jpgfile)))
            
            im, w, h, nw, nh, iw, ih = letterbox_image(im, tuple(reversed(model_image_size)))
            image_data = np.array(im, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)
            # 计算梯度
            # with graph.as_default():
            cost, gradients, out_classes, out_boxes = self.sess.run(
                [cost_function, gradient_function, self.classes, self.boxes],
                feed_dict={
                    self.yolo9_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            e_x = e * (0.9 ** index)
            print("batch:{} Cost: {:.8}".format(index, cost))
            for i, c in reversed(list(enumerate(out_classes))):
                box = out_boxes[i]
                top, left, bottom, right = box
                top_list.append(top)
                left_list.append(left)
                bottom_list.append(bottom)
                right_list.append(right)
            if not top_list and index == 0: # 原始图像没有目标且列表为空
                top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
            else:
                top_min = min(top_list)
                left_min = min(left_list)
                bottom_max = max(bottom_list)
                right_max = max(right_list)
            # 计算运行时间
            # start = timer()
            # 计算噪声

            if attack_name == 'SMGM':
                pre_n = np.sign(pre_g)
                g = gradients
                n = np.sign(g)
                pre_g = g
                image_data -= (pre_n * e + n * e)
                image_data = np.clip(image_data, 0, 1.0)
            index += 1
            if cost < 0.000002:
                break
            # end = timer()
            # time = end - start
            # time_sum += time
        # log_folder = os.path.join("logs/{}".format(dataset))
        # if not os.path.exists(log_folder):
            # os.makedirs(log_folder)
        # with open('logs/{}/{}_time.txt'.format(dataset, attack_name), 'a', encoding='utf-8') as f:
            # f.write(str("{}".format(time_sum)) + '\n')
        # print(time_sum)
        # time_sum = 0
        return 0

    def TargetAttack(self, image, attack_name, dataset, count, jpgfile, model_image_size=(640, 640)):
        # start = timer()
        global graph
        ori_image = image # 扰动截取之用
        pixdata_1 = ori_image.load()
        
        boxed_image, w, h, nw, nh, iw, ih = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # 增加额外维度
        image_data = np.expand_dims(image_data, 0)
        original_image = np.copy(image_data)
        out_scores, out_classes = self.sess.run(
            [self.scores, self.classes],
            feed_dict={
                self.yolo9_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # 定义maxcost
        maxcost = 0
        linshi_cost = 1
        time_sum = 0
        object_hack = 2  # 该参数为要攻击的目标类别
        object_target = 5  # 该参数为攻击成的目标类别
        A = self.classes >= object_target
        B = self.classes <= object_target
        hack_scores = tf.boolean_mask(self.scores, A & B)
        cost_function = tf.reduce_sum(hack_scores)  # 跑通代码，定向攻击
        gradient_function = K.gradients(cost_function, self.yolo9_model.input)[0]
        cost = 1
        e = 4 / 255
        n = 0
        r = 0
        index = 0
        top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
        top_list= []
        left_list= []
        bottom_list= []
        right_list= []
        # 最大改变幅度
        max_change_above = original_image + 0.3
        max_change_below = original_image - 0.3
        # 初始设置
        pre_g = np.zeros(image_data.shape)
        gradients_m = np.zeros(image_data.shape)
        # 主要攻击循环
        # if object_hack in list(out_classes):
        # while linshi_cost > 0.3:
        for i in range(0, 20):
            img = image_data[0]
            img *= 255.
            im = Image.fromarray(img.astype(np.uint8))
            im = letterbox_image_1(im, w, h, nw, nh)
            im = im.resize((iw, ih), Image.BICUBIC)
            '''裁剪扰动'''
            if attack_name == 'SMGM':
                # im = im.convert("RGB")
                pixdata = im.load()
                for i_width in range(iw):#遍历图片的所有像素
                    for j_height in range(ih):
                        if i_width < left_min or i_width > right_max or j_height < top_min or j_height > bottom_max:
                            pixdata[i_width,j_height] = pixdata_1[i_width,j_height]
            '''临时用'''
            im.save(os.path.join("mAP//input", os.path.basename(jpgfile)))
            # 再次打开图片
            im = Image.open(os.path.join("mAP//input", os.path.basename(jpgfile)))

            im, w, h, nw, nh, iw, ih = letterbox_image(im, tuple(reversed(model_image_size)))
            image_data = np.array(im, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)
            # 计算梯度
            cost, gradients, out_scores, out_classes, out_boxes = self.sess.run(
                [cost_function, gradient_function, self.scores, self.classes, self.boxes],
                feed_dict={
                    self.yolo9_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0})
            for j, c in reversed(list(enumerate(out_classes))):
                score = out_scores[j]
                if c == 2:
                    maxcost += score
                box = out_boxes[j]
                top, left, bottom, right = box
                top_list.append(top)
                left_list.append(left)
                bottom_list.append(bottom)
                right_list.append(right)
            print(maxcost)
            if not top_list and index == 0: # 原始图像没有目标且列表为空
                top_min, left_min, bottom_max, right_max = 0, 0, 0, 0 # 初始化坐标
            else:
                top_min = min(top_list)
                left_min = min(left_list)
                bottom_max = max(bottom_list)
                right_max = max(right_list)
            # start = timer()
            # 计算噪声
            if attack_name == 'SMGM':
                pre_n = np.sign(pre_g)
                g = gradients
                n = np.sign(g)
                pre_g = g
                image_data += (pre_n * e + n * e)
                image_data = np.clip(image_data, 0, 1.0)

            print("batch:{} Cost: {:.8}".format(index, cost))
            # with open('logs/PGD.txt', 'a', encoding='utf-8') as f:
            #     f.write(str("batch:{} Cost: {:.8}%".format(index, cost * 100)) + '\n')
            index += 1
            # if maxcost < 0.3:
                # break
            linshi_cost = maxcost
            maxcost = 0
            # end = timer()
            # time = end - start
            # time_sum += time
        # with open('logs/{}_time.txt'.format(attack_name), 'a', encoding='utf-8') as f:
            # f.write(str("{}".format(time_sum)) + '\n')
        # time_sum = 0
        return 0


if __name__ == '__main__':
    model_path = 'yolo9_weights.h5'
    anchors_path = 'model_data/yolo9_anchors.txt'
    classes_path = 'model_data/coco_classes.txt'
    score = 0.5
    iou = 0.5
    model_image_size = (640, 640)

    import glob
    dataset = 'voc'  # 'COCO'  'voc'
    path = "test/voc-test2007/original/*.jpg"
    attack_names = ["SMGM"]
    for attack_name in attack_names:
        count = 0
        for jpgfile in glob.glob(path):
            yolo9_model = Yolo9(score, iou, anchors_path, classes_path, model_path)
            img = Image.open(jpgfile)
            result = yolo9_model.Attack_1(img, attack_name, dataset, count, jpgfile, model_image_size=model_image_size)
            # result = yolo9_model.TargetAttack_1(img, attack_name, dataset, count, jpgfile, model_image_size=model_image_size)
            count += 1
            print(count)
            K.clear_session()
        yolo9_model.close_session()


    # import glob
    # # path = "mAP/input/targeted/COCO/AM4-FGSM/19/*.jpg"
    # path = "output/*.jpg"
    # outdir = "test/result"
    # yolo9_model = Yolo9(score, iou, anchors_path, classes_path, model_path)
    # for jpgfile in glob.glob(path):
        # img = Image.open(jpgfile)
        # # with open('train.txt', 'a', encoding='utf-8') as f:
            # # f.write(("{}".format(jpgfile)) + ' ')
        # img, out_classes = yolo9_model.detect_image(img)
        # # with open('train.txt', 'a', encoding='utf-8') as f:
            # # f.write('\n')
        # img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    # yolo9_model.close_session()

