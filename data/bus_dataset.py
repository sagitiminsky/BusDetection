""" COCO dataset (quick and dirty)

Hacked together by Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import numpy as np
from PIL import Image
import pickle



def load_dataset_from_pickle(pickle_files, transform=None):
    data = pickle.load(open(pickle_files, 'rb'))
    image_data = data[0]
    box_data = data[1]
    class_data = data[2]
    return BusDataSet(image_data, box_data, class_data, transform=transform)


class BusDataSet(data.Dataset):
    def __init__(self, image_data, box_data, class_data, transform=None):
        super(BusDataSet, self).__init__()
        self.transform = transform

        self.image_data = image_data
        self.box_data = box_data
        self.class_data = class_data

        self.n_samples = len(self.image_data)

    def _parse_img_ann(self, index):
        bboxes = []
        cls = []
        image_size = self.image_data[index].shape[:2]
        box_data = self.box_data[index]
        class_data = self.class_data[index]
        for box, class_index in zip(box_data, class_data):
            bboxes.append(box)
            cls.append(class_index)

        bboxes = np.array(bboxes, dtype=np.float32)
        cls = np.array(cls, dtype=np.int64)

        return dict(img_id=index, bbox=bboxes, cls=cls, img_size=image_size)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """

        ann = self._parse_img_ann(index)
        img = Image.fromarray(self.image_data[index]).convert('RGB')
        if self.transform is not None:
            img, ann = self.transform(img, ann)

        return img, ann

    def __len__(self):
        return self.n_samples

    def split(self, split_size):

        end_list = [0]
        ds_list = []
        for s in split_size:
            start_index = end_list[-1]
            n = round(self.n_samples * s)
            end_index = n + start_index
            end_list.append(end_index)
            print(start_index, end_index)
            ds_list.append(BusDataSet(self.image_data[start_index:end_index], self.box_data[start_index:end_index],
                                      self.class_data[start_index:end_index], transform=self.transform))
        return ds_list


if __name__ == '__main__':
    ds = BusDataSet('/data/datasets/BusProject/pacal_data.pickle')
    img, ann = ds[0]
    print("a")
