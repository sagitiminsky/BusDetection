import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import pickle

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def read_xml(file_path):
    tree = ET.parse(file_path)
    object_list = []
    folder = None
    filename = None
    size = None
    for i in tree.getroot():
        if i.tag == 'folder':
            folder = i.text
        if i.tag == 'filename':
            filename = i.text
        if i.tag == 'size':
            size = tuple([int(j.text) for j in i.getchildren()])
        if i.tag == 'object':
            class_name = [j.text for j in i.getchildren() if j.tag == 'name']
            for j in i.getchildren():
                if j.tag == 'bndbox':
                    bbox = {n.tag: float(n.text) for n in j.getchildren()}

            # bbox = [[float(n.text) for n in j.getchildren()] for j in i.getchildren() if j.tag == 'bndbox']
            object_list.append(
                np.asarray([bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax'], CLASSES.index(class_name[0])]))

    return object_list, folder, filename, size


def read_dataset_labels(labels_folder, only_class='bus'):
    xml_files = os.listdir(labels_folder)
    dataset_label_dict = dict()
    class_filter = CLASSES.index(only_class)
    for xml_file in xml_files:
        object_list, folder, filename, size = read_xml(os.path.join(labels_folder, xml_file))
        object_list = [obj for obj in object_list if obj[-1] == class_filter]
        if len(object_list) > 0:
            dataset_label_dict.update({filename: (object_list, folder, size)})
    return dataset_label_dict


if __name__ == '__main__':
    label_path = '/data/datasets/BusProject/PascalVOC2012/VOC2012/Annotations/'
    image_data_folder = '/data/datasets/BusProject/PascalVOC2012/VOC2012/JPEGImages'
    dataset_label_dict = read_dataset_labels(label_path)
    image_list = []
    bbox_list = []
    cls_list = []
    for file_name, v in dataset_label_dict.items():
        file_path = os.path.join(image_data_folder, file_name)
        image = Image.open(file_path)
        image_list.append(np.asarray(image))

        bbox_list.append([i[:4] for i in v[0]])
        cls_list.append([1 for i in v[0]])
    pickle.dump((image_list, bbox_list, cls_list), open('/data/datasets/BusProject/pacal_data.pickle', 'wb'))
