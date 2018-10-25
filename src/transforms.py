import torch
import random
import numpy as np
import pandas as pd

import cv2

cv2.setNumThreads(0)


def img_size(image: np.ndarray):
    return (image.shape[1], image.shape[0])


def image_cumsum(img):
    cumsum_img = np.float32(img).cumsum(axis=0)
    cumsum_img -= cumsum_img.min()
    if cumsum_img.max() > 0:
        cumsum_img /= cumsum_img.max()
    cumsum_img *= 255
    cumsum_img = np.clip(cumsum_img, 0, 255)
    cumsum_img = cumsum_img.astype(np.uint8)
    return cumsum_img


def get_samples_for_aug(folds_path):
    images_lst = []
    train_folds_df = pd.read_csv(folds_path)

    for i, row in train_folds_df.iterrows():
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) == 0:
            image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
            images_lst.append(image)
    print("Images for background augmentation:", len(images_lst))
    return images_lst


def img_crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]


def random_crop(img, size):
    tw = size[0]
    th = size[1]
    w, h = img_size(img)
    if ((w - tw) > 0) and ((h - th) > 0):
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
    else:
        x1 = 0
        y1 = 0
    img_return = img_crop(img, (x1, y1, x1 + tw, y1 + th))
    return img_return, x1, y1


class ProbOutputTransform:
    def __init__(self, segm_thresh=0.5, prob_thresh=0.5, size=None):
        self.segm_thresh = segm_thresh
        self.prob_thresh = prob_thresh
        self.crop = None
        if size is not None:
            self.crop = CenterCrop(size)

    def __call__(self, preds):
        segms, probs = preds
        preds = segms > self.segm_thresh
        probs = probs > self.prob_thresh
        preds = preds * probs.view(-1, 1, 1, 1)
        if self.crop is not None:
            preds = self.crop(preds)
        return preds


class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, image, mask=None):
        if mask is None:
            if random.random() < self.prob:
                transform = random.choice(self.transforms)
                image = transform(image)
            return image
        else:
            if random.random() < self.prob:
                transform = random.choice(self.transforms)
                image, mask = transform(image, mask)
            return image, mask


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, mask=None):
        if mask is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, mask = self.transform(image, mask)
            return image, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        if mask is None:
            for trns in self.transforms:
                image = trns(image)
            return image
        else:
            for trns in self.transforms:
                image, mask = trns(image, mask)
            return image, mask


class GaussNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, img, mask=None):
        if self.sigma_sq > 0.0:
            img = self._gauss_noise(img,
                                    np.random.uniform(0, self.sigma_sq))
        if mask is None:
            return img
        return img, mask

    def _gauss_noise(self, img, sigma_sq):
        img = img.astype(np.uint32)
        w, h, c = img.shape
        gauss = np.random.normal(0, sigma_sq, (h, w))
        gauss = gauss.reshape(h, w)
        img = img + np.stack([gauss for i in range(c)], axis=2)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        return img


class SpeckleNoise:
    def __init__(self, sigma_sq):
        self.sigma_sq = sigma_sq

    def __call__(self, img, mask=None):
        if self.sigma_sq > 0.0:
            img = self._speckle_noise(img,
                                      np.random.uniform(0, self.sigma_sq))
        if mask is None:
            return img
        return img, mask

    def _speckle_noise(self, img, sigma_sq):
        sigma_sq /= 255
        img = img.astype(np.uint32)
        w, h, c = img.shape
        gauss = np.random.normal(0, sigma_sq, (h, w))
        gauss = gauss.reshape(h, w)
        img = img + np.stack([gauss for i in range(c)], axis=2) * img
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        return img


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img, x1, y1 = random_crop(img, self.size)
        if mask is None:
            return img
        mask = img_crop(mask, (x1, y1, x1 + self.size[0],
                               y1 + self.size[1]))
        return img, mask


class Flip:
    def __init__(self, flip_code):
        assert flip_code in [0, 1]
        self.flip_code = flip_code

    def __call__(self, img, mask=None):
        img = cv2.flip(img, self.flip_code)
        if mask is None:
            return img
        mask = cv2.flip(mask, self.flip_code)
        return img, mask


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class RandomBg:
    def __init__(self, input_path, max_w):
        # TODO Update if not use the SimpleDepthTransform
        self.bg_imgs = get_samples_for_aug(input_path)
        assert max_w < 1.0, "Weight of the augmentet background should be < 1.0"
        self.max_w = max_w

    def __call__(self, img, mask=None):
        w = np.random.uniform(0, self.max_w)
        n = np.random.randint(0, len(self.bg_imgs))
        bg = self.bg_imgs[n]
        img = cv2.addWeighted(img, 1.0 - w, np.stack([bg, bg, bg], axis=2), w, 0)
        if mask is None:
            return img
        return img, mask


class PadToSize:
    def __init__(self, size, random_center=False, mode='reflect'):
        self.size = np.array(size, int)
        self.random_center = random_center
        self.mode = mode

    def __call__(self, img, mask=None):
        diff = self.size - img.shape[:2]
        assert np.all(diff >= 0)

        if self.random_center:
            pad_left = []
            for i in range(len(diff)):
                if diff[i]:
                    pad_left.append(np.random.randint(0, diff[i]))
                else:
                    pad_left.append(0)
        else:
            pad_left = diff // 2
        pad_right = diff - pad_left
        pad_width = ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]))
        if self.mode == 'constant':
            pad_img = np.pad(img, pad_width + ((0, 0),), self.mode, constant_values=0)
        else:
            pad_img = np.pad(img, pad_width + ((0, 0),), self.mode)
        if mask is None:
            return pad_img
        pad_mask = np.pad(mask, pad_width, self.mode)
        return pad_img, pad_mask


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask=None):
        w, h = img.shape[1], img.shape[0]
        tw, th = self.size
        if w == tw and h == th:
            if mask is None:
                return img
            else:
                return img, mask

        x1 = (w - tw) // 2
        y1 = (h - th) // 2

        crop_img = img_crop(img, (x1, y1, x1 + tw, y1 + th))

        if mask is None:
            return crop_img
        crop_mask = img_crop(mask, (x1, y1, x1 + tw, y1 + th))
        return crop_img, crop_mask


class RandomHorizontalScale:
    def __init__(self, max_scale, interpolation=cv2.INTER_NEAREST):
        assert max_scale >= 1.0, "RandomHorizontalScale works as upscale only"
        self.max_scale = max_scale
        self.interpolation = interpolation

    def __call__(self, image, mask=None):
        if self.max_scale > 1.0:
            scale_factor = np.random.uniform(1.0, self.max_scale)
            resized_image = self._scale_img(image, scale_factor)
            if mask is None:
                return resize_image
            resized_mask = self._scale_img(mask, scale_factor)
            return resized_image, resized_mask
        else:
            if mask is None:
                return image
            return image, mask

    def _scale_img(self, img, factor):
        h, w = img.shape[:2]
        new_w = round(w * factor)
        img = cv2.resize(img, dsize=(new_w, h),
                         interpolation=self.interpolation)
        if new_w > w:
            random_x = np.random.randint(0, new_w - w)
        else:
            random_x = 0
        return img[:, random_x:(random_x + w)]


class Blur:
    def __init__(self, ksize=(5, 1), sigma_x=20):
        self.ksize = ksize
        self.sigma_x = sigma_x

    def __call__(self, image, mask=None):
        blured_image = cv2.GaussianBlur(image, self.ksize, self.sigma_x)
        if mask is None:
            return blured_image
        return blured_image, mask


class Scale:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, image, mask=None):
        resize_image = cv2.resize(image, self.size, interpolation=self.interpolation)
        if mask is None:
            return resize_image
        resize_mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return resize_image, resize_mask


class SimpleDepthTransform:
    def __call__(self, image, depth=None):
        input = np.stack([image, image, image], axis=2)
        return input


class DepthTransform:
    def __call__(self, image, depth):
        cumsum_img = image_cumsum(image)
        depth_img = np.full_like(image, depth // 4)
        input = np.stack([image, cumsum_img, depth_img], axis=2)
        return input


class ImageToTensor:
    def __init__(self, coord_channels=False):
        self.coord_channels = coord_channels

    def add_coord_channels(self, image_tensor):
        _, h, w = image_tensor.size()
        for row, const in enumerate(np.linspace(0, 1, h)):
            image_tensor[1, row, :] = const
        for col, const in enumerate(np.linspace(0, 1, w)):
            image_tensor[2, :, col] = const
        return image_tensor

    def __call__(self, image):
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        if self.coord_channels:
            image = self.add_coord_channels(image)
        return image


class MaskToTensor:
    def __call__(self, mask):
        mask[mask > 0] = 1
        return mask.astype(np.float32)[np.newaxis]


class SaltToTensor:
    def __init__(self, coord_channels=False):
        self.img2tensor = ImageToTensor(coord_channels)
        self.msk2tensor = MaskToTensor()

    def __call__(self, image, mask=None):
        if mask is None:
            return self.img2tensor(image)
        return self.img2tensor(image), self.msk2tensor(mask)


class Dummy:
    def __call__(self, image, mask=None):
        return image, mask


class SaltTransform:
    def __init__(self, size, train, resize_mode='pad', coord_channels=False):
        assert resize_mode in ['scale', 'pad', 'crop']
        self.train = train

        if train:
            if resize_mode == 'scale':
                resize = Scale(size)
            elif resize_mode == 'pad':
                resize = OneOf([
                    PadToSize(size, random_center=True),
                    PadToSize(size, random_center=False)
                ], prob=1)
            elif resize_mode == 'crop':
                resize = RandomCrop(size)
            else:
                raise Exception
            self.transform = Compose([
                resize,
                UseWithProb(HorizontalFlip(), 0.5),
                SaltToTensor(coord_channels)
            ])
        else:
            if resize_mode == 'scale':
                resize = Scale(size)
            elif resize_mode == 'pad':
                resize = PadToSize(size, random_center=False)
            elif resize_mode == 'crop':
                resize = CenterCrop(size)
            else:
                raise Exception
            self.transform = Compose([
                resize,
                SaltToTensor(coord_channels)
            ])

    def __call__(self, image, mask=None):
        if mask is None:
            image = self.transform(image)
            return image
        else:
            image, mask = self.transform(image, mask)
            return image, mask
