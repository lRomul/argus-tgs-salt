import numpy as np
import pandas as pd
import cv2
import os
from os.path import join
from collections import defaultdict

from src.utils import filename_without_ext
from src.config import TRAIN_DIR, TEST_DIR

IMAGE_SHAPE = (101, 101)

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_WHITE = (255,255,255)

def stack_images(images_array):
    img_rows = []
    for row in images_array:
        img_row = []
        for img in row:
            img_row.append(img)
        img_rows.append(np.concatenate(img_row, axis=1))
    return np.concatenate(img_rows, axis=0)


def split_mosaic(mosaic):
    img_array = []
    for row_img in np.split(mosaic, mosaic.shape[0]//101, axis=0):
        img_row_array = []
        for img in np.split(row_img, mosaic.shape[1]//101, axis=1):
            img_row_array.append(img)
        img_array.append(img_row_array)
    return img_array


class Mosaic:
    def __init__(self, raw_mosaic):

        i_min = min([p['i'] for p in raw_mosaic])
        j_min = min([p['j'] for p in raw_mosaic])

        i_max = -np.inf
        j_max = -np.inf

        mosaic = []
        for raw_piece in raw_mosaic:
            piece = dict()
            piece['i'] = raw_piece['i'] - i_min
            piece['j'] = raw_piece['j'] - j_min
            piece['id'] = raw_piece['id']
            mosaic.append(piece)

            if piece['i'] > i_max:
                i_max = piece['i']
            if piece['j'] > j_max:
                j_max = piece['j']

        id2index = dict()
        mosaic_array = np.full((j_max + 1, i_max + 1), fill_value=None, dtype=object)
        for piece in mosaic:
            mosaic_array[piece['j'], piece['i']] = piece['id']
            id2index[piece['id']] = piece['j'], piece['i']

        self.pieces = mosaic
        self.array = mosaic_array
        self.id2index = id2index
        self.ids = set(self.id2index.keys())

    def get_neighbor(self, id, i_shift, j_shift):
        assert id in self.id2index, f'Mosaic have not piece with id {id}'

        j, i = self.id2index[id]
        neig_j = j + j_shift
        neig_i = i + i_shift
        if neig_j < 0 or neig_j >= self.array.shape[0]:
            return None
        if neig_i < 0 or neig_i >= self.array.shape[1]:
            return None
        return self.array[neig_j, neig_i]

    def get_neighbors(self, id):
        assert id in self.id2index, f'Mosaic have not piece with id {id}'

        return np.array(np.matrix([
            [self.get_neighbor(id, -1, -1), self.get_neighbor(id, 0, -1), self.get_neighbor(id, 1, -1)],
            [self.get_neighbor(id, -1, 0), id, self.get_neighbor(id, 1, 0)],
            [self.get_neighbor(id, -1, 1), self.get_neighbor(id, 0, 1), self.get_neighbor(id, 1, 1)]
        ]))

    def __contains__(self, item):
        return item in self.id2index

    def __repr__(self):
        string = ""
        for j in range(self.array.shape[0]):
            for i in range(self.array.shape[1]):
                string += f'{self.array[j, i]} '
            string += '\n'
        return string
        
    def __len__(self):
        return len(self.pieces)


class Mosaics:
    def __init__(self, mosaic_csv_path):
        self.mosaics_csv_path = mosaic_csv_path

        mosaic_df = pd.read_csv(mosaic_csv_path)
        raw_mosaic_dict = defaultdict(list)

        for i, row in mosaic_df.iterrows():
            raw_mosaic_dict[row.mosaic_id].append({
                'i': row.x,
                'j': row.y,
                'id': row.id
            })

        self.mosaic_id2mosaic = dict()
        self.id2mosaic_id = dict()
        self.id2mosaic = dict()
        self.ids = set()
        for mosaic_id, raw_mosaic in raw_mosaic_dict.items():
            mosaic = Mosaic(raw_mosaic)
            for id in mosaic.ids:
                self.id2mosaic_id[id] = mosaic_id
                self.id2mosaic[id] = mosaic
            self.mosaic_id2mosaic[mosaic_id] = mosaic
            self.ids.update(mosaic.ids)

    def __contains__(self, id):
        return id in self.ids

    def __getitem__(self, mosaic_id):
        return self.mosaic_id2mosaic[mosaic_id]


class SaltData:
    def __init__(self,
                 train_dir=TRAIN_DIR,
                 test_dir=TEST_DIR,
                 pred_dir=None,
                 images_dir_name='images',
                 masks_dir_name='masks',
                 mosaic_csv_path=None):
        test_images_dir = join(test_dir, images_dir_name)
        test_images_names = os.listdir(test_images_dir)
        self.pred_dir = pred_dir
        train_images_dir = join(train_dir, images_dir_name)
        train_masks_dir = join(train_dir, masks_dir_name)
        train_images_names = os.listdir(train_images_dir)
        self.empty_image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        self.full_image = np.full(IMAGE_SHAPE, fill_value=255, dtype=np.uint8)

        img_paths = [join(test_images_dir, name) for name in test_images_names]
        img_paths += [join(train_images_dir, name) for name in train_images_names]
        mask_paths = [join(train_masks_dir, name) for name in train_images_names]
        self.id2pred = {}
        if self.pred_dir is not None:
            pred_images_names = os.listdir(self.pred_dir)
            pred_mask_paths = [join(self.pred_dir, name)
                               for name in pred_images_names]
            self.id2pred = {filename_without_ext(pred_mask_path):
                            cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                            for pred_mask_path in pred_mask_paths}
        self.id2image = {filename_without_ext(img_path): cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                         for img_path in img_paths}
        self.id2mask = {filename_without_ext(mask_path): cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        for mask_path in mask_paths}
        self.id2pred_cor = {}
        self.train_ids = set([filename_without_ext(n) for n in train_images_names])
        self.test_ids = set([filename_without_ext(n) for n in test_images_names])

        # Remove empty images from train
        for id in list(self.train_ids):
            image = self.id2image[id]
            if not np.sum(image):
                del self.id2image[id]
                self.train_ids.remove(id)

        self.ids = self.train_ids | self.test_ids

        self.mosaics = mosaic_csv_path
        if mosaic_csv_path is not None:
            self.mosaics = Mosaics(mosaic_csv_path)

    def in_train(self, id):
        assert id in self.ids
        return id in self.train_ids

    def in_mosaics(self, id):
        assert id in self.ids
        return id in self.mosaics

    def get_neighbors(self, id):
        if id in self.mosaics:
            mosaic = self.mosaics.id2mosaic[id]
            return mosaic.get_neighbors(id)
        else:
            return np.array(np.matrix([
                [None, None, None],
                [None, id, None],
                [None, None, None],
            ]))

    def add_pred_cor(self, id, pred_cor):
        self.id2pred_cor[id] = pred_cor

    def draw_visualization(self, id_array, draw_names=False):
        img_array = []
        for row in id_array:
            img_row = []
            for id in row:
                if id is None:
                    img = np.stack([self.empty_image]*3, axis=2)
                else:
                    img = self.id2image[id].astype(int)
                    img = np.stack([img] * 3, axis=2)
                    if id in self.id2mask:
                        img[:, :, 1] += (self.id2mask[id] * 0.25).astype(int)
                        img[:, :, 2] += 25
                    if id in self.id2pred:
                        img[:, :, 0] += (self.id2pred[id] * 0.25).astype(int)
                    if id in self.id2pred_cor:
                        img[:, :, 0] += (self.id2pred_cor[id] * 0.25).astype(int)
                        img[:, :, 1] += (self.id2pred_cor[id] * 0.25).astype(int)
                    img = np.clip(img, 0, 255)
                img = img.astype(np.uint8)
                if draw_names:
                    cv2.putText(img, str(id), (5, 60),
                                FONT, 0.45, COL_WHITE, 1)
                img_row.append(img.astype(np.uint8))
            img_array.append(img_row)
        return stack_images(img_array)

    def get_stacked_images(self, id_array, return_unknown_mask=False):
        img_array = []
        unknown_mask_array = []
        for row in id_array:
            img_row = []
            unknown_mask_row = []
            for id in row:
                if id is None:
                    img = self.empty_image
                    unknown_mask = self.full_image
                else:
                    img = self.id2image[id]
                    unknown_mask = self.empty_image
                img_row.append(img)
                unknown_mask_row.append(unknown_mask)
            img_array.append(img_row)
            unknown_mask_array.append(unknown_mask_row)
        if return_unknown_mask:
            return stack_images(img_array), stack_images(unknown_mask_array)
        return stack_images(img_array)

    def get_stacked_masks(self, id_array, return_unknown_mask=False):
        mask_array = []
        unknown_mask_array = []
        for row in id_array:
            mask_row = []
            unknown_mask_row = []
            for id in row:
                if id is None:
                    mask = self.empty_image
                    unknown_mask = self.full_image
                elif id in self.id2mask:
                    # In train
                    mask = self.id2mask[id]
                    unknown_mask = self.empty_image
                else:
                    # In test
                    mask = self.empty_image
                    unknown_mask = self.full_image
                mask_row.append(mask)
                unknown_mask_row.append(unknown_mask)
            mask_array.append(mask_row)
            unknown_mask_array.append(unknown_mask_row)
        if return_unknown_mask:
            return stack_images(mask_array), stack_images(unknown_mask_array)
        return stack_images(mask_array)

    def get_pred_stacked_masks(self, id_array, return_unknown_mask=False):
        mask_array = []
        unknown_mask_array = []
        for row in id_array:
            mask_row = []
            unknown_mask_row = []
            for id in row:
                if id is None:
                    mask = self.empty_image
                    unknown_mask = self.full_image
                elif id in self.id2mask:
                    # In train
                    mask = self.id2mask[id]
                    unknown_mask = self.empty_image
                else:
                    # In test
                    mask = self.id2pred[id]
                    unknown_mask = self.full_image
                mask_row.append(mask)
                unknown_mask_row.append(unknown_mask)
            mask_array.append(mask_row)
            unknown_mask_array.append(unknown_mask_row)
        if return_unknown_mask:
            return stack_images(mask_array), stack_images(unknown_mask_array)
        return stack_images(mask_array)

    def get_pred_masks(self, id_array, return_unknown_mask=False):
        mask_array = []
        unknown_mask_array = []
        for row in id_array:
            mask_row = []
            unknown_mask_row = []
            for id in row:
                if id is None:
                    mask = self.empty_image
                    unknown_mask = self.full_image
                elif id in self.id2mask:
                    # In train
                    mask = self.id2mask[id]
                    unknown_mask = self.empty_image
                else:
                    # In test
                    mask = self.id2pred[id]
                    unknown_mask = self.full_image
                mask_row.append(mask)
                unknown_mask_row.append(unknown_mask)
            mask_array.append(mask_row)
            unknown_mask_array.append(unknown_mask_row)
        if return_unknown_mask:
            return mask_array, unknown_mask_array
        return mask_array
