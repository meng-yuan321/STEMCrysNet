import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from torch_geometric.data import Data
import pickle
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import re
import warnings
from stemcrysnet.common.utils import PROJECT_ROOT
from stemcrysnet.common.data_utils import LMDBDataset
import cv2
from scipy.ndimage import gaussian_filter
import random


class CrystLMDBDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, two_stem: ValueNode, switch: ValueNode, is_training: ValueNode, add_noise: ValueNode, rotate1: ValueNode = True, rotate2: ValueNode = True,   
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.two_stem = two_stem
        self.switch = switch
        self.is_training = is_training
        self.add_noise = add_noise
        self.rotate1 = rotate1
        self.rotate2 = rotate2
        self.cached_data = LMDBDataset(path)
        
    def cut_off(self, n):
        data_list = []
        for data in tqdm(self.cached_data):
            if data['atoms_num'] <= n:
                data_list.append(data)
        self.cached_data = data_list

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        filename = data_dict['filename']
        # 提取id
        # id = 1
        pattern = re.compile(r'[-_](\d+)')
 
        match = pattern.search(filename)
        if match:
            id = int(match.group(1))
        else:
            warnings.warn(f"Filename '{filename}' does not match the expected pattern. Using index as fallback id.")
            id = int(index)
    
        frac_coords = np.array(data_dict['atom_frac_pos'])
        atom_types = np.array(data_dict['atom_type'])
        num_atoms = (data_dict['atoms_num'])
        lengths = np.array(data_dict['parameters'][:3])
        angles = np.array(data_dict['parameters'][3:])

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            id=id,
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            idx=index,
            num_atoms=num_atoms,
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        img = np.array(data_dict['img']) # （2，770，770）
        peak = np.random.uniform(150,500) # (30,300) 
        std = np.random.uniform(0,5) #(0,20)
        crop_size = (256, 256)
        h, w = img.shape[1:]
        top = np.random.randint(0, h - crop_size[0] + 1)
        left1 = np.random.randint(0, w - crop_size[1] + 1)
        left2 = np.random.randint(0, w - crop_size[1] + 1)

        # all_ps_list = [0.003128091, 0.006256181, 0.003753709, 0.007507418, 0.004692136, 0.009384272, 0.012512363, 0.007820227, 0.015640453]
        # ps = random.choice(all_ps_list)
        ps = None
        if self.two_stem:
            if len(img.shape) == 3:
                new_img = self._augment(img[0, :, :], rotate=self.rotate1, rotate90=True, ps=ps, top=top, left=left1, crop_size=crop_size)
                if self.add_noise:
                    new_img = self._add_noise(new_img, peak, std)
                new_img = self._normalize(new_img)
                data.stem_img = torch.as_tensor(new_img) # （256，256）

                new_img = self._augment(img[1, :, :], rotate=self.rotate2, rotate90=True, ps=ps, top=top, left=left2, crop_size=crop_size)
                if self.add_noise:
                    new_img = self._add_noise(new_img, peak, std)
                new_img = self._normalize(new_img)
                data.stem_img_yz = torch.as_tensor(new_img)

                
                if self.switch and (index % 2 == 0): # 随机交换两个视角的位置
                    data.stem_img, data.stem_img_yz = data.stem_img_yz, data.stem_img
                    
                # import matplotlib.pyplot as plt
                # os.makedirs("img_noise_testloader", exist_ok=True)
                # plt.imsave(f"img_noise_testloader/{filename}_0.png", data.stem_img.numpy(), cmap='gray')
                # plt.imsave(f"img_noise_testloader/{filename}_1.png", data.stem_img_yz.numpy(), cmap='gray')
            else:
                raise ValueError(f"len(img.shape) != 3.")
        else:
            if len(img.shape) == 3:
                # img = self._augment(img[0, :, :], rotate=True, ps=ps)
                if self.switch and (index % 2 == 0):
                    img = self._augment(img[1, :, :], rotate=True, rotate90=False, ps=ps, crop_size=crop_size)
                else:
                    img = self._augment(img[0, :, :], rotate=True, rotate90=False, ps=ps, crop_size=crop_size)

                if self.add_noise:
                    img = self._add_noise(img, peak, std)
                img = self._normalize(img)
                # import matplotlib.pyplot as plt
                # os.makedirs("img_noise_testloader_one", exist_ok=True)
                # plt.imsave(f"img_noise_testloader_one/{filename}.png", img, cmap='gray')
                data.stem_img = torch.as_tensor(img)
            else:
                raise ValueError(f"len(img.shape) != 3.")
        return data
    def _normalize(self, image):
        return (image-image.min())/(image.max()-image.min()+1.0e-10) 
    
    def _normalizeAndConvert2uint8(self, image):
        return (((image - image.min())/image.ptp()) * np.iinfo(np.uint8).max).astype(np.uint8) if image.ptp() != 0 else image.astype(np.uint8)

    def _augment(self, image, rotate, rotate90, ps=None, top=0, left=0, crop_size=(256,256)):
        
        # if ps is not None:
        #     image = self._resize_image(image, 0.0075, ps)
        if rotate:
            angle = np.random.randint(0, 360)
            image = self._rotate_image(image, angle)
            if random.random() > 0.5:
                image = np.flipud(image)
            if random.random() > 0.5:
                image = np.fliplr(image)
        if rotate90:
            image = np.rot90(image, 1) # 确保行是对应的，适配sttr
            if random.random() > 0.5:
                image = np.fliplr(image)
        
        
        if rotate:
            image = self._crop_center_image(image)
        else:
            image = self._crop_random_image(image, top, left, crop_size)

        sigma = np.random.uniform(1, 4)
        image = gaussian_filter(image, sigma=sigma)
        return image
    
    def _resize_image(self, image, ps, target_ps):
        scale = ps/target_ps
        # target_size = int(h*scale), int(w*scale)
        return cv2.resize(image, dsize=(0,0), fx=scale, fy=scale) 
    
    def _rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        return image
    
    def _crop_center_image(self, image):
        crop_size = (256,256)
        max_offset = 10
        h, w = image.shape[:2]

        center_top = (h - crop_size[0]) // 2
        center_left = (w - crop_size[1]) // 2

        top_min = max(0, center_top - max_offset)
        top_max = min(h - crop_size[0], center_top + max_offset)
        left_min = max(0, center_left - max_offset)
        left_max = min(w - crop_size[1], center_left + max_offset)

        top = np.random.randint(top_min, top_max + 1)
        left = np.random.randint(left_min, left_max + 1)
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        return image[top:bottom, left:right]
    
    def _crop_random_image(self, image, top, left, crop_size):
        bottom = top + crop_size[0]
        right = left + crop_size[1]

        return image[top:bottom, left:right]

    def _add_noise(self, image, peak, std):
        # sigma = 3 #np.random.uniform(1, 5)
        # image = gaussian_filter(image, sigma=sigma)
        
        image_poisson = self._add_poisson_noise(image, peak)
        
        image_poisson_gaussian = self._add_gaussian_noise(image_poisson, std = std)
        # optionally add scan-direction noise (random angle)
        # if random.random() < 0.5:
        #     try:
        #         image_poisson_gaussian = self.add_scan_noise(image_poisson_gaussian)
        #     except Exception:
        #         # fallback to the gaussian-poisson result on any error
        #         pass
        return image_poisson_gaussian
    
    def _add_poisson_noise(self, image_array, peak=30):
        # image_array = np.asarray(image).astype(np.float32)
        noisy_image_array = np.random.poisson(image_array / 255.0 * peak) / peak * 255.0
        noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
        return noisy_image_array

    def _add_gaussian_noise(self, image_array, mean=0, std=30):
        # image_array = np.asarray(image).astype(np.float32)
        noise = np.random.normal(mean, std, image_array.shape)
        noisy_image_array = image_array + noise
        noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
        return noisy_image_array


    def __repr__(self) -> str:
        return f"CrystMOFLMDBDataset({self.name=}, {self.path=})"


