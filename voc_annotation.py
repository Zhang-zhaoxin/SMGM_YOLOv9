import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ['person', 'bicycle', 'car', 'motorcycle', 'aeroplane', 'bus',
#                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#                 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#                 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
#                 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
#                 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(round(float(xmlbox.find('xmin').text))), int(round(float(xmlbox.find('ymin').text))), int(round(float(xmlbox.find('xmax').text))),
             int(round(float(xmlbox.find('ymax').text))))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
