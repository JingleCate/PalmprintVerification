import os
import cv2
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as tf

from PIL import Image


def split(root:
          str = r"C:\Users\21552\Desktop\Main\Courses\AI\ComputerVision\projects\sourcecode\ROI"):
    """save the train and test path in .txt file for convenient process

    Args:
        root (str, optional): Defaults to "./ROI".
    """

    train_txt = "./datapaths/train_path.txt"
    valid_txt = "./datapaths/valid_path.txt"
    test_txt = "./datapaths/test_path.txt"

    # Image path list
    img_files = [os.path.join(root, f)
                 for f in os.listdir(root)
                 if os.path.isfile(os.path.join(root, f))]

    division1 = int(len(img_files)*0.8)       # 8:1:1
    division2 = int(len(img_files)*0.9) 
    train_img_files = img_files[0:division1]
    valid_img_files = img_files[division1:division2]
    test_img_files = img_files[division2:]

    # print("path: %s\n person: %s\n hand: %s\n" %  path1, path2, bt_hand))

    # Train, validation and test path txt file
    with open(train_txt, 'w') as f_train:
        for item in train_img_files:
            # _ = path, fullname = abc.py
            _, fullname = os.path.split(item)
            name, _ = os.path.splitext(fullname)        # name = abc
            person = name[0:3]      # such as 001,002
            hand = name[8:9]        # left hand or right hand
            f_train.write("%s %s %s\n" % (item, person, hand))
            
    with open(valid_txt, 'w') as f_test:
        for item in test_img_files:
            # _ = path, fullname = abc.py
            _, fullname = os.path.split(item)
            name, _ = os.path.splitext(fullname)        # name = abc
            person = name[0:3]      # such as 001,002
            hand = name[8:9]        # left hand or right hand
            f_test.write("%s %s %s\n" % (item, person, hand))

    with open(test_txt, 'w') as f_test:
        for item in test_img_files:
            # _ = path, fullname = abc.py
            _, fullname = os.path.split(item)
            name, _ = os.path.splitext(fullname)        # name = abc
            person = name[0:3]      # such as 001,002
            hand = name[8:9]        # left hand or right hand
            f_test.write("%s %s %s\n" % (item, person, hand))


def created_composed_dataset(path, saved_path: str = "./datapaths/train_resampled.txt", shuffle: bool = False):
    data, subitem = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')     # drop the '\n'
            subitem = line.split()
            data.append(subitem)
    size = len(data)
    i = 0
    index = []
    random.seed(0)
    while i * 80 < size:
        # Totaly sample the 400*40 same sample
        # each 80 items contain 40 left-hand images and 40 right-hand images of same person
        left_hand, right_hand = [], []
        for j in range(0, 4):
            for k in range(0, 10):
                left_hand.append(80*i+20*j+k)
                right_hand.append(80*i+10+20*j+k)

        # Random select 40 samples each hand
        for j in range(0, 40):
            random_list1 = random.sample(left_hand, 2)
            random_list2 = random.sample(right_hand, 2)
            index.append(random_list1)
            index.append(random_list2)
            print("Same: %d-%d" % (i, j))
            # print("random_list1: ", random_list1)
            # print("random_list2: ", random_list2)
        i += 1

    for j in range(size):
        # print("Diff: %d" % (i))
        index.append([random.randint(0, size-1), random.randint(0, size-1)])

    if shuffle == True:
        random.shuffle(index)
    true = 0
    false = 0
    div = len(index)//2
    k = 0
    while k < div:
        index[k+1], index[div+k+1] = index[div+k+1], index[k+1]
        index[k+3], index[div+k+3] = index[div+k+3], index[k+3]
        k += 4

    with open(saved_path, 'w') as f:
        for i in range(len(index)):
            it1, it2 = index[i][0], index[i][1]
            # print("it1: %d it2: %d" % (it1, it2))
            path1, person1, hand1 = data[it1][0], data[it1][1], data[it1][2]
            path2, person2, hand2 = data[it2][0], data[it2][1], data[it2][2]
            target = 1 if person1 == person2 and hand1 == hand2 else 0
            if target == 1:
                true += 1
            else:
                false += 1

            f.write("%s %s %s\n" % (path1, path2, target))
    print("True samples: %d False samples: %d" % (true, false))


def proprecess(path1: str, path2: str, device, batch_size):
    transform = tf.Compose([
            tf.ToPILImage(),
            tf.Resize(224),
            tf.ToTensor(),
            tf.Normalize([1, 1, 1], [1, 1, 1])
        ])
    processed_batch_data = torch.tensor([])
    for i in range(batch_size):
        img1 = cv2.imread(path1[i])
        # Transform the image
        
        img1 = transform(img1)

        # Reshape the image. PyTorch model reads 4-dimensional tensor
        # [batch_size, channels, width, height]
        img1 = img1.reshape(1, 3, 224, 224)
        img1 = img1.to(device)

        img2 = cv2.imread(path2[i])
        img2 = transform(img2)

        img2 = img2.reshape(1, 3, 224, 224)
        img2 = img2.to(device)

        cated_tensor = torch.cat((img1, img2), dim=1)
        # print("Cat: ", cated_tensor.shape)
        processed_batch_data = torch.cat(
            (processed_batch_data, cated_tensor), dim=0)

    return processed_batch_data


class CustomDataset(data.Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data_list = self.read_data_path()

        true = 0
        false = 0
        for item in self.data_list:
            if item[2] == '0':
                false += 1
            elif item[2] == '1':
                true += 1
        print("Dataset -- Same samples: %d, different samples: %d" %
              (true, false), end="  ")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path1, path2, target = self.data_list[index][0], self.data_list[index][1], self.data_list[index][2]

        return path1, path2, target

    def read_data_path(self):
        data, subitem = [], []
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip('\n')     # drop the '\n'
                subitem = line.split()
                data.append(subitem)
        return data


if __name__ == "__main__":
    # split()
    # created_composed_dataset("./datapaths/train_path.txt", saved_path="./datapaths/train_resampled.txt", shuffle=True)
    # created_composed_dataset("./datapaths/valid_path.txt", saved_path="./datapaths/valid_resampled.txt", shuffle=True)
    created_composed_dataset("./datapaths/test_path.txt", saved_path="./datapaths/test_resampled.txt", shuffle=True)
    pass
