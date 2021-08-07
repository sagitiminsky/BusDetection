import numpy as np
import cv2
import random
import math
import os
import ast
from matplotlib import pyplot as plt
import uuid
import os, shutil

# resources
# https://www.kaggle.com/nvnnghia/awesome-augmentation
# https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

imgdir = "test/busesTest"

finalSize=1280
THRESHOLD = 100

### DO NOT CAHNGE ###
targetSize = finalSize*2

mosaic_root="mosaic/busesTrain"
d = {}
for line in open('test/annotationsTest.txt'):
    key, value = line.split(':')
    d[key] = np.atleast_2d(ast.literal_eval(value))



def delete_folder_content():
    folder = mosaic_root
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def show_resized():
    number_of_images = 5
    selected = []
    fig, ax = plt.subplots(1, number_of_images, figsize=(20, 10))
    for i in range(number_of_images):
        index = random.choice([x for x in range(len(os.listdir(imgdir))) if x not in selected])  # select a random image
        selected.append(index)
        selected_image = os.listdir(imgdir)[index]
        sample, bboxes, (h, w) = load_image(selected_image)

        for box in bboxes:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            cv2.rectangle(sample,
                          (x, y),
                          (x + w, y + h),
                          (255, 0, 0), 3)

        ax[i].imshow(sample)
    plt.show()


def load_image(imagename):
    """loads image, resizes it and it's bounding boxes"""

    img = cv2.imread(f'{imgdir}/{imagename}', cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    assert img is not None, 'Image Not Found:' + f'{imgdir}/{imagename}'

    # resize to img_size//2
    resized = targetSize // 2
    img = cv2.resize(img, (resized, resized))

    # normalize bbox to targetSize
    y_, x_ = img.shape[:2]

    x_scale = x_ / w
    y_scale = y_ / h

    bboxes_ret=[]

    for bbox in d[imagename]:
        x_ = bbox[0]
        y_ = bbox[1]
        w_ = bbox[2]
        h_ = bbox[3]
        color=bbox[4]

        x = int(np.round(x_ * x_scale))
        y = int(np.round(y_ * y_scale))
        w = int(np.round(w_ * x_scale))
        h = int(np.round(h_ * y_scale))


        bboxes_ret.append([x,y,w,h,color])


    return img, np.array(bboxes_ret), img.shape[:2]  # img,bboxes, hw


def load_mosaic():
    """loads images in a mosaic"""

    labels4 = []
    xc, yc = [int(random.uniform(targetSize, targetSize)) for _ in range(2)]  # mosaic center x, y
    padw, padh = targetSize//2, targetSize//2
    selected = []
    indices = []
    for _ in range(4):
        index = random.choice([x for x in range(len(os.listdir(imgdir))) if x not in selected])
        indices.append(index)
        selected.append(index)

    img4 = np.full((targetSize, targetSize, 3), 255, dtype=np.uint8)  # base image with 4 tiles

    for i, index in enumerate(indices):
        # Load image
        selected_image = os.listdir(imgdir)[index]
        img, bboxes, (h, w) = load_image(selected_image)

        # place img in img4
        if i == 0:  # top left

            # Image
            img4[:targetSize // 2, :targetSize // 2, :] = img  # img4[ymin:ymax, xmin:xmax]

            # Labels
            labels = bboxes.copy()
            if bboxes.size > 0:  # Normalized xywh to pixel xyhw format
                labels[:, 0] = bboxes[:, 0]
                labels[:, 1] = bboxes[:, 1]
                labels[:, 2] = bboxes[:, 2]
                labels[:, 3] = bboxes[:, 3]
                labels[:, 4] = bboxes[:, 4]  # bus color



        elif i == 1:  # top right

            # Labels
            labels = bboxes.copy()
            if bboxes.size > 0:  # Normalized xywh to pixel xyhw format
                labels[:, 0] = bboxes[:, 0] + padw
                labels[:, 1] = bboxes[:, 1]
                labels[:, 2] = bboxes[:, 2]
                labels[:, 3] = bboxes[:, 3]
                labels[:, 4] = bboxes[:, 4]  # bus color

            # Image
            img4[:targetSize // 2, targetSize // 2:, :] = img  # img4[ymin:ymax, xmin:xmax]

        elif i == 2:  # bottom left

            # Image
            img4[targetSize // 2:, :targetSize // 2, :] = img  # img4[ymin:ymax, xmin:xmax]

            # Labels
            labels = bboxes.copy()
            if bboxes.size > 0:  # Normalized xywh to pixel xyhw format
                labels[:, 0] = bboxes[:, 0]
                labels[:, 1] = bboxes[:, 1] + padh
                labels[:, 2] = bboxes[:, 2]
                labels[:, 3] = bboxes[:, 3]
                labels[:, 4] = bboxes[:, 4]  # bus color



        elif i == 3:  # bottom right

            # Image
            img4[targetSize // 2:, targetSize // 2:, :] = img  # img4[ymin:ymax, xmin:xmax]

            # Labels
            labels = bboxes.copy()
            if bboxes.size > 0:  # Normalized xywh to pixel xyhw format
                labels[:, 0] = bboxes[:, 0] + padw
                labels[:, 1] = bboxes[:, 1] + padh
                labels[:, 2] = bboxes[:, 2]
                labels[:, 3] = bboxes[:, 3]
                labels[:, 4] = bboxes[:, 4]  # bus color

        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)


    return img4, labels4


def show_samples():
    delete_folder_content()
    number_of_images = 700
    # fig, ax = plt.subplots(1, number_of_images, figsize=(20, 10))
    with open("{}/annotationGT.txt".format(mosaic_root), 'a') as the_file:

        for i in range(number_of_images):
            sample, boxes = load_mosaic()

            image=sample[targetSize // 4:3 * targetSize // 4, targetSize // 4:3 * targetSize // 4]
            cropped = image.copy()
            augmented_bboxes=[]
            for box in boxes:


                x, delta_x = magic_clip(int(box[0]))
                y, delta_y = magic_clip(int(box[1]))
                x2 = x+int(box[2]) if delta_x == 0 else int(box[2] - delta_x)
                y2 = y+int(box[3]) if delta_y == 0 else int(box[3] - delta_y)

                w=abs(np.clip(x2,0,targetSize//2)-x)
                h=abs(np.clip(y2,0,targetSize//2)-y)


                if w*h>THRESHOLD:

                    augmented_bboxes.append([x,y,w,h,box[4]])
                    cv2.rectangle(cropped,
                                  (x, y),
                                  (np.clip(x2,0,targetSize//2), np.clip(y2,0,targetSize//2)),
                                  (255, 0, 0), 2)

            # ax[i].imshow(cropped)


            if augmented_bboxes:
                image_name = str(uuid.uuid4())
                cv2.imwrite("{}/{}.JPG".format(mosaic_root, image_name), image)
                the_file.write("{}.JPG:{}\n".format(image_name, str([[int(value) for value in box] for box in augmented_bboxes])[1:-1]))




    # plt.show()


def magic_clip(a):
    delta = 0
    if int(a) - targetSize // 4 < 0:
        delta = abs(int(a) - targetSize // 4)
        return np.clip(int(a) - targetSize // 4, 0, targetSize // 2), delta



    return np.clip(int(a) - targetSize // 4, 0, targetSize//2), delta



if __name__ == "__main__":
    show_samples()
    # show_resized()
