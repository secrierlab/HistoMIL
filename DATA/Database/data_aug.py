#############################################################################
import numpy as np
import numpy.random as np_rand

from PIL import Image, ImageFilter

from skimage import color
import skimage
from torchvision import transforms

from HistoMIL.DATA.Database.stain_norm import naive_normalize,StainNormalizer


"""
predefined data augmentation for Histopathology

"""
class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        if np_rand.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = np_rand.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = img.rotate(random_rotation*90)
        return img


class RandomHEStain(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """
    def __init__(self,scale:float=0.01,center:float = 1.0):
        self.scale = scale
        self.center = center

    def __call__(self, img):
        img_he = skimage.color.rgb2hed(img)
        img_he[:, :, 0] = img_he[:, :, 0] * np_rand.normal(self.center, self.scale, 1)  # H
        img_he[:, :, 1] = img_he[:, :, 1] * np_rand.normal(self.center, self.scale, 1)  # E
        img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
        img = Image.fromarray(np.uint8(img_rgb*254.999))
        return img


class RandomGaussianNoise(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """

    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(np_rand.normal(0.0, 0.5, 1)))
        return img


class NaiveHistoNormalize(object):
    """Normalizes the given PIL.Image"""

    def __call__(self, img):
        img_arr = np.array(img)
        img_norm = naive_normalize(img_arr)
        img = Image.fromarray(img_norm)
        return img

class HistoNormalize(object):
    """Good normalise function can select normalise approach"""

    def __init__(self, method_name:str="vahadane",paras:dict={}):
        self.normalizer = StainNormalizer()
        # select normaliser from ["reuifrok","macenko", "vahadane","custom","reinhard"]
        self.normalizer.set_worker(method_name=method_name)
        self.normalizer.fit(target=paras["target"],paras=paras)
        self.paras = paras

    def __call__(self, img):
        img_arr = np.array(img)
        img_norm = self.normalizer.transform(img_arr,self.paras)
        img = Image.fromarray(img_norm)
        return img



"""
predefined data augmentation for ssl

acturally implement split_transform() for dataloader
"""
import attr
import torch
import torchvision.transforms as transforms
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

@attr.s(auto_attribs=True)
class SSL_DataAug:
    resize: int = 256
    s: float = 0.5
    apply_blur: bool = True

    def split_trs_fn(self, img) -> torch.Tensor:
        transform = self.single_transform()
        return torch.stack((transform(img), transform(img)))

    def test_trs_fn(self, img) -> torch.Tensor:
        transform = self.get_test_transform()
        return transform(img)

    def single_transform(self):
        transform_list = [
            RandomHEStain(),
            #NaiveHistoNormalize(),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
        ]
        if self.apply_blur:
            transform_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        
        transform_list.append(NaiveHistoNormalize())
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def get_test_transform(self):
        return transforms.Compose(
            [
                #transforms.CenterCrop(self.crop_size),
                #transforms.ToTensor(),
                NaiveHistoNormalize(),
                transforms.ToTensor(),
            ]
        )

    def get_trans_fn(self,is_train:bool=False):
        if is_train:
            return self.split_trs_fn
        else:
            return self.test_trs_fn






def no_transforms(is_train=False):

    return transforms.Compose([
                            #----> for  image net normalization
                            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            transforms.ToTensor(),
                            ])


def only_naive_transforms(is_train=False):
    return transforms.Compose([
                            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                            NaiveHistoNormalize(),
                            transforms.ToTensor(),
                            ])


def naive_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
                                RandomHEStain(),
                                NaiveHistoNormalize(),
                                RandomRotate(),
                                RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                ])
    else:
        return transforms.Compose([
                                NaiveHistoNormalize(),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])



