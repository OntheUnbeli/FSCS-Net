import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.utils import color_map
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale


class SUNRGBD(data.Dataset):

    def __init__(self, cfg, mode='train', do_aug=True):
        assert mode in ['train', 'test']

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']
        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            Resize(crop_size),
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        self.val_resize = Resize(crop_size)
        self.class_weight = np.array([4.1489, 4.7033, 6.0432, 21.9638, 19.1237, 11.3568, 24.0552, 15.1227,
            23.6196, 22.7505, 39.8595, 38.5019, 36.8810, 42.1598, 27.4841, 42.7398,
            34.5860, 40.8405, 39.3031, 39.7861, 49.9436, 43.8466, 34.9577, 43.7357,
            45.1824, 46.2051, 43.9607, 46.3582, 49.9228, 40.8575, 39.0468, 48.2333,
            48.3639, 44.6668, 42.9215, 45.1362, 45.8617, 45.8814])
        # self.class_weight = np.array([0.0522, 0.0608, 0.0824, 0.3970, 0.2748, 0.1827, 0.3923, 0.2697, 0.5148,
        #     0.4961, 0.6812, 1.6903, 0.9299, 0.9536, 0.5529, 1.2164, 0.7504, 0.9379,
        #     1.5654, 1.1292, 1.5259, 1.8584, 1.0512, 2.4692, 0.8776, 1.7570, 3.1422,
        #     2.4001, 1.3498, 2.0586, 0.7029, 4.3062, 2.4925, 1.0799, 1.5464, 3.9556,
        #     0.8348, 3.7300])
        # self.class_weight = np.array([0.0522,0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
        #            0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
        #            2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
        #            0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
        #            1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
        #            4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
        #            3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
        #            0.750738, 4.040773])

        self.mode = mode
        self.do_aug = do_aug

        self.train_ids = list(range(5285))  # 5285 - last-587 = 4698
        self.test_ids = list(range(5050))

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        else:
            return len(self.test_ids)

    def __getitem__(self, index):
        if self.mode == 'train':
            image_index = self.train_ids[index]
        else:
            image_index = self.test_ids[index]

        image_path = f'{self.mode}/image/{image_index}.jpg'
        depth_path = f'{self.mode}/depth/{image_index}.png'
        label_path = f'{self.mode}/label/{image_index}.png'
        # label_path = '/home/yangenquan/PycharmProjects/SUN_RGBD/test/label/234.png'
        image = Image.open(os.path.join(self.root, image_path))  # RGB 0~255
        depth = Image.open(os.path.join(self.root, depth_path)).convert('RGB')  # 1 channel -> 3
        label = Image.open(os.path.join(self.root, label_path))  # 1 channel 0~37


        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.mode == 'train' and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)
        else:
            sample = self.val_resize(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = label_path.strip().split('/')[-1]  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return color_map(N=self.n_classes)


if __name__ == '__main__':
    import json

    path = '/home/yangenquan/PycharmProjects/第一论文模型/(60.1)mymodel8/configs/sunrgbd.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)

    # cfg['root'] = '/home/dtrimina/Desktop/lxy/database/SUNRGBD'
    dataset = SUNRGBD(cfg, mode='test')
    from toolbox.utils import class_to_RGB
    import matplotlib.pyplot as plt
    print(len(dataset))
    for i in range(len(dataset)):
        sample = dataset[i]

        image = sample['image']
        # print(image.shape, 'iamge')
        depth = sample['depth']
        # print(depth.shape, 'depth')
        label = sample['label']
        # print(i, set(label.cpu().reshape(-1).tolist()), 'label')
        # print(image.shape)
        # mask = label > 0
        # target_m = label.clone()
        # target_m[mask] -= 1
        # print(i, set(target_m.cpu().reshape(-1).tolist()), 'label')
        image = image.numpy()
        image = image.transpose((1, 2, 0))
        image *= np.asarray([0.229, 0.224, 0.225])
        image += np.asarray([0.485, 0.456, 0.406])

        depth = depth.numpy()
        depth = depth.transpose((1, 2, 0))
        # depth *= np.asarray([9650, 9650, 9650])
        # depth += np.asarray([19050, 19050, 19050])
        depth *= np.asarray([0.226, 0.226, 0.226])
        depth += np.asarray([0.449, 0.449, 0.449])

        label = label.numpy()
        # print(dataset.cmap)
        label = class_to_RGB(label, N=38, cmap=dataset.cmap)
        # print(dataset.cmap)

        # plt.subplot('131')
        # plt.imshow(image)
        # plt.subplot('132')
        # plt.imshow(depth)
        # plt.subplot('133')
        # plt.imshow(label)
        # plt.show()
        label = Image.fromarray(label)
        # plt.imshow(depth)
        label.save('/home/yangenquan/PycharmProjects/SUN_RGBD/test_color/{}.jpg'.format(i))





