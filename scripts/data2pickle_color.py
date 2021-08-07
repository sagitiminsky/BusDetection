import numpy as np
import os
from PIL import Image
import pickle

image_data_folder = '/data/datasets/BusProject/busesTrain'
anno_file = '/data/datasets/BusProject/annotationsTrain.txt'

if __name__ == '__main__':
    file1 = open(anno_file, 'r')
    lines = file1.readlines()
    dataset_label_dict = dict()
    for l in lines:
        file_name, labels = l.split(':')
        label_array = np.stack([np.fromstring(b.split('[')[1], np.uint64, sep=',') for b in labels.split(']')[:-1]])
        label_array_new = []
        for instance in label_array:
            x_min = instance[0]
            y_min = instance[1]
            w = instance[2]
            h = instance[3]
            x_max = x_min + w
            y_max = y_min + h
            class_index = instance[4]
            label_array_new.append([y_min, x_min, y_max, x_max, class_index])

        dataset_label_dict[file_name] = np.asarray(label_array_new)

    image_list = []
    bbox_list = []
    cls_list = []
    for file_name, v in dataset_label_dict.items():
        file_path = os.path.join(image_data_folder, file_name)
        image = Image.open(file_path)
        image_list.append(np.asarray(image))

        bbox_list.append([i[:4].astype('float') for i in v])
        cls_list.append([i[-1] for i in v])


    os.makedirs('/data/datasets/BusProject/', exist_ok=True) # should be already created manually
    pickle.dump((image_list, bbox_list, cls_list), open('/data/datasets/BusProject/color_data.pickle', 'wb'))
