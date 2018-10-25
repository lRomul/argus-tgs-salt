import os
import cv2
import json
import numpy as np
from os.path import join
from shutil import copyfile
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import linemerge, unary_union, polygonize

from src.utils import filename_without_ext

from mosaic.mosaic_api import SaltData
from mosaic.math_fit import fit_third_order

MOSAIC_PATH = '/workdir/data/mosaic/pazzles_6013.csv'


left = 0
right = 1
top = 2
bottom = 3
img_side_len = 101

BIG_NUMBER = 1000
SMALL_NUMBER = 1e-3

def find_points(mask, x_shift=0, y_shift=0):
    # Find points where mask change class on edges
    mask = mask > 0
    mask = mask.astype(np.int)
    n = mask.shape[1]
    edges = [mask[:, 0+x_shift], mask[:, -1-x_shift],
             mask[0+y_shift, :], mask[-1-y_shift, :]]
    diffs = [np.diff(edge, n=1) for edge in edges]
    pos = [np.argwhere(diff>0)+1 for diff in diffs]
    neg = [np.argwhere(diff<0)+1 for diff in diffs]
    pos = [[int(x) for x in p] for p in pos]
    neg = [[int(x) for x in n] for n in neg]
    if mask[0, 0] > 0:
        for i in [left, top]:
            pos[i] = [0] + pos[i]
    if mask[-1, 0] > 0:
        pos[bottom] = [0] + pos[bottom]
        neg[left] = [n] + neg[left]
    if mask[0, -1] > 0:
        pos[right] = [0] + pos[right]
        neg[top] = [n] + neg[top]
    if mask[-1, -1] > 0:
        for i in [right, bottom]:
            neg[i] = [n] + neg[i]
    return(pos, neg)

def is_closed(pos_s, neg_s):
    return len(pos_s) == 1 and pos_s[0] == 0\
            and len(neg_s) == 1 and neg_s[0] == img_side_len

def is_filled(pos, neg):
    return all([is_closed(p, n) for (p, n) in zip(pos, neg)])

def is_empty(mask):
    return np.max(mask) == 0

def is_side_closed(pos, neg, side):
    return is_closed(pos[side], neg[side])

def len_side_closed(mask, side):
    if side == left:
        s = mask[:, 0]
    elif side == right:
        s = mask[:, -1]
    elif side == top:
        s = mask[0, :]
    else:
        s = mask[-1, :]
    return np.count_nonzero(s)

def is_terminal(mask):
    # Check if the given mask is a terminal tile
    return len_side_closed(mask, top) < len_side_closed(mask, bottom)/2

def is_lines(pos, neg):
    # Check if the given mask contains only verlical lines
    return pos[top] == pos[bottom] and neg[top] == neg[bottom]

def can_be_matched(top_tile, bottom_tile):
    # Check if two masks can be connected with lines
    top_tile_pos, top_tile_neg = top_tile
    bottom_tile_pos, bottom_tile_neg = bottom_tile
    return len(bottom_tile_pos[top]) == len(top_tile_pos[bottom])\
            and len(bottom_tile_neg[top]) == len(top_tile_neg[bottom])


def tile_diff(tile, idx, side, pos_type=True, depth=2):
    # Calculate derivative on tiles edges
    pos, neg = find_points(tile)
    pos_crop, neg_crop = find_points(tile, x_shift=depth, y_shift=depth)
    if side in [left, right]:
        dx = depth
        if pos_type:
            if len(pos_crop[side]) != len(pos[side]):
                return SMALL_NUMBER
            dy = pos_crop[side][idx] - pos[side][idx]
        else:
            if len(neg_crop[side]) != len(neg[side]):
                return SMALL_NUMBER
            dy = neg_crop[side][idx] - neg[side][idx]
    else:
        dy = depth
        if pos_type:
            if len(pos_crop[side]) != len(pos[side]):
                return BIG_NUMBER
            dx = pos_crop[side][idx] - pos[side][idx]
        else:
            if len(neg_crop[side]) != len(neg[side]):
                return BIG_NUMBER
            dx = neg_crop[side][idx] - neg[side][idx]
    if dx == 0:
        return SMALL_NUMBER
    else:
        return dy/dx

def poly_3(a, b, c, d, x):
    return x*(x*(a*x+b)+c)+d

def can_be_corner(tile1, tile2, left):
    '''Check if two tiles in a rectangular pattern can have a missed corner between
    
    The tiles relation (left=False):
    
    |corn?|tile2| Where corn? - examined tile. 
    |tile1|-----| The scheme is reflected in case left=True
    '''
    pos1, neg1 = find_points(tile1)  # points of the bottom-center tile
    pos2, neg2 = find_points(tile2)  # points of the top-side tile
    ret = None
    if left:
        x_match = img_side_len + 1
        k_x = 0
        for k, x in enumerate(neg1[top]):
            if 0 < x < img_side_len and x < x_match:
                x_match = x
                k_x = k
        y_match = -1
        k_y = 0
        for k, y in enumerate(pos2[right]):
            if 0 < y < img_side_len and y > y_match:
                y_match = y
                k_y = k
        if x_match != img_side_len + 1 and y_match != -1:
            d1 = tile_diff(tile1, k_x, top)
            d2 = tile_diff(tile2, k_y, right)
            ret = [(img_side_len, x_match, d1), (y_match, 0, d2)]
    else:
        x_match = -1
        k_x = 0
        for k, x in enumerate(pos1[top]):
            if 0 < x < img_side_len and x > x_match:
                x_match = x
                k_x = k
        y_match = -1
        k_y = 0
        for k, y in enumerate(pos2[left]):
            if 0 < y < img_side_len and y > y_match:
                y_match = y
                k_y = k
        if x_match != -1 and y_match != -1:
            d1 = tile_diff(tile1, k_x, top)
            d2 = tile_diff(tile2, k_y, left)
            ret = [(img_side_len, x_match, d1),
                   (y_match, img_side_len, d2)]
    return ret


def pair_len(pair):
    return pair[1] - pair[0] - 1

def draw_v_pair(pair, strip, l):
    # Drow polligons between two masks from test
    if l > 0:
        roi_points = [(0, 0), (img_side_len, 0),
                      (img_side_len*l, img_side_len), (0, img_side_len)]
        roi_poly = Polygon(roi_points)
        top_tile = find_points(pair[0])
        bottom_tile = find_points(pair[1])
        top_tile_pos, top_tile_neg = top_tile
        bottom_tile_pos, bottom_tile_neg = bottom_tile
        v_shift = l * img_side_len

        square_points = [(0, 0), (img_side_len, 0), (img_side_len, v_shift), (0, v_shift)]
        square_poly = Polygon(square_points)
        lines = []
        for i in range(len(top_tile_pos[bottom])):
            line = LineString([(top_tile_pos[bottom][i], 0),
                               (bottom_tile_pos[top][i], v_shift),
                               (bottom_tile_neg[top][i], v_shift),
                               (top_tile_neg[bottom][i], 0)])
            lines.append(line)

        merged = linemerge([square_poly.boundary, *lines])
        borders = unary_union(merged)
        polygons = []
        for poly in polygonize(borders):
            polygons.append(poly)
        masks = [mask_for_polygons([polygons[i]], (v_shift, img_side_len))
                 for i in range(0, len(polygons), 2)]
        mask = (np.any(masks, axis=0)*255).astype(np.uint8)
        return mask
    return None

def divide(mask, ni, nj):
    # Divide a mask into 101x101 px tiles
    step_i = mask.shape[0]//ni
    step_j = mask.shape[1]//nj
    crops = [[[] for _j in range(nj)] for _i in range(ni)]
    for i in range(ni):
        for j in range(nj):
            crop = mask[i*step_i:(i+1)*step_i,
                        j*step_j:(j+1)*step_j]
            crops[i][j] = crop
    return crops

# From https://michhar.github.io/masks_to_polygons_and_back/
def mask_for_polygons(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 255)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def find_fit_corners(tile1, tile2, left):
    # Draw missed corners polygons
    c = can_be_corner(tile1, tile2, left)
    ret = None
    if c is not None:
        x1 = c[0][0]
        x2 = c[1][0]
        y1 = c[0][1]
        y2 = c[1][1]
        pol = fit_third_order(x1, x2, y1, y2, c[0][2], c[1][2])
        if pol is not None:
            pol_points = []
            if left:
                pol_points.append((min(y1,y2), min(x1, x2)))
            else:
                pol_points.append((min(y1,y2), max(x1, x2)))
            pol_points_a = []
            for x_i in range(min(x1, x2), max(x1, x2)+1):
                y_i = int(poly_3(*pol, x_i))
                pol_points_a.append((int(y_i), int(x_i)))
            if all([0<=x[0]<=img_side_len for x in pol_points_a])\
                    and all([0<=x[1]<=img_side_len for x in pol_points_a]):
                pol_points.extend(pol_points_a)
            if left:
                pol_points.append((max(y1, y2), max(x1, x2)))
                pol_points.append((img_side_len, 0))
            else:
                pol_points.append((max(y1,y2), min(x1, x2)))
                pol_points.append((img_side_len, img_side_len))
            
            square_points = [(0, 0), (img_side_len, 0),
                             (img_side_len, img_side_len),
                             (0, img_side_len)]
            square_poly = Polygon(square_points)
            line = LineString(pol_points)

            merged = linemerge([square_poly.boundary, line])
            borders = unary_union(merged)
            polygons = []
            for poly in polygonize(borders):
                polygons.append(poly)
            if len(polygons) > 1:
                return mask_for_polygons([polygons[1]],
                                         (img_side_len, img_side_len))

def fix_holes(mask):
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    _, contour,hier = cv2.findContours(thresh, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 255, -1)
    return mask

def postprocess(mean_prediction_dir, postprocessed_prediction_dir):
    '''Make the postprocessing
    
    Args:
        mean_prediction_dir (str): Path to a dir with mean predictions
        postprocessed_prediction_dir (str): Path to save postprocessed tiles
    '''

    def find_pairs(strip):
        # Return a list of indexes of train images pairs in a vertical strip
        pairs = []
        opening = None
        for i, s in enumerate(strip):
            if s[0] in saltdata.id2mask:
                if opening is not None:
                    pairs.append((opening, i))
                opening = i
        return pairs

    def pair_match(strip, pairs):
        # Return a list of matched pair images from train
        new_pairs = []
        for pair in pairs:
            top_tile = find_points(saltdata.id2mask[strip[pair[0]][0]])
            bottom_tile = find_points(saltdata.id2mask[strip[pair[1]][0]])
            if can_be_matched(top_tile, bottom_tile):
                new_pairs.append(pair)
        return new_pairs

    def find_pairs_pred(strip):
        # Return a list of indexes of train images pairs in a vertical strip
        pairs = []
        opening = None
        for i, s in enumerate(strip):
            if s[0] in saltdata.id2pred\
                    and np.count_nonzero(saltdata.id2pred[s[0]])>0:
                if opening is not None:
                    pairs.append((opening, i))
                opening = i
        return pairs

    def pair_match_pred(strip, pairs):
        # Return a list of matched pair images from train
        new_pairs = []
        for pair in pairs:
            if strip[pair[0]][0] in saltdata.id2pred_cor:
                top_tile = find_points(saltdata.id2pred_cor[strip[pair[0]][0]])
            else:
                top_tile = find_points(saltdata.id2pred[strip[pair[0]][0]])
            if strip[pair[1]][0] in saltdata.id2pred_cor:
                bottom_tile = find_points(saltdata.id2pred_cor[strip[pair[1]][0]])
            else:
                bottom_tile = find_points(saltdata.id2pred[strip[pair[1]][0]])
            if can_be_matched(top_tile, bottom_tile):
                new_pairs.append(pair)
        return new_pairs
    
    def process_strip(strip, term_level=-1):
        # Postprocess a vertical column of tiles
        pairs = pair_match(strip, find_pairs(strip))
        for pair in pairs:
            empty = [is_empty(saltdata.id2mask[strip[k][0]]) for k in pair]
            l = pair_len(pair)
            if l > 0:
                ids = [strip[i][0] for i in range(pair[0]+1, pair[1])]
                if not any(empty):
                    mask = draw_v_pair((saltdata.id2mask[strip[pair[0]][0]],
                                        saltdata.id2mask[strip[pair[1]][0]]), strip, l)
                    pred_cors = divide(mask, l, 1)
                    for i in range(l):
                        saltdata.add_pred_cor(ids[i], pred_cors[i][0])

        can_be_copied = [-1] * len(strip)
        for i, tile in enumerate(strip):
            tile_id = tile[0]
            if tile_id in saltdata.id2mask:
                tile_mask = saltdata.id2mask[tile_id]
                if is_lines(*find_points(tile_mask))\
                        and not is_empty(tile_mask):
                    for j in range(i+1, len(strip)):
                        if strip[j][0] not in saltdata.id2pred_cor:
                            can_be_copied[j] = i
        for i in range(len(can_be_copied)):
            if can_be_copied[i] >= 0:
                saltdata.add_pred_cor(strip[i][0], saltdata.id2mask[strip[can_be_copied[i]][0]])

        can_be_copied_up = [-1] * len(strip)
        for i in range(len(strip)-1, -1, -1):
            tile_id = strip[i][0]
            if tile_id in saltdata.id2mask:
                tile_mask = saltdata.id2mask[tile_id]
                if is_lines(*find_points(tile_mask))\
                        and not is_empty(tile_mask):
                    for j in range(i-1, term_level, -1):
                        is_term = False
                        j_id = strip[j][0]
                        if j_id in saltdata.id2pred and not is_empty(saltdata.id2pred[j_id]):
                            is_term = is_terminal(saltdata.id2pred[j_id])
                        elif j_id in saltdata.id2mask:
                            is_term = True
                        if j_id not in saltdata.id2pred_cor and not is_term:
                            can_be_copied_up[j] = i
                        else:
                            break

        for i in range(len(can_be_copied_up)):
            if can_be_copied_up[i] >= 0:
                saltdata.add_pred_cor(strip[i][0], saltdata.id2mask[strip[can_be_copied_up[i]][0]])
        pairs = pair_match_pred(strip, find_pairs_pred(strip))
        for pair in pairs:
            l = pair_len(pair)
            if l > 0:
                ids = [strip[i][0] for i in range(pair[0]+1, pair[1])]
                not_in_masks = [id not in saltdata.id2mask for id in ids]
                if all(not_in_masks):
                
                    m1 = get_tile_mask(strip[pair[0]][0])
                    m2 = get_tile_mask(strip[pair[1]][0])
                    if m1 is not None and m2 is not None:
                        mask = draw_v_pair((m1, m2), strip, l)
                        pred_cors = divide(mask, l, 1)
                        for i in range(l):
                            if ids[i] not in saltdata.id2mask:
                                saltdata.add_pred_cor(ids[i], pred_cors[i][0])

    def zero_under_salt(strip, term_level):
        # Set all tiles to zero under completely filled bottom edge of a tile
        can_be_filled = [False] * len(strip)
        for i in range(term_level, len(strip)):
            tile_id = strip[i][0]
            if tile_id in saltdata.id2mask:
                tile_mask = get_tile_mask(tile_id)
                if is_side_closed(*find_points(tile_mask), bottom)\
                        and not is_empty(tile_mask):
                    for j in range(i+1, len(strip)):
                        can_be_filled[j] = True
    
        for i in range(len(can_be_filled)):
            fill_id = strip[i][0]
            if can_be_filled[i] and fill_id in saltdata.id2pred:
                saltdata.add_pred_cor(fill_id, np.zeros((img_side_len, img_side_len), dtype=np.uint8))
            elif can_be_filled[i] and fill_id in saltdata.id2pred_cor:
                saltdata.add_pred_cor(fill_id, np.zeros((img_side_len, img_side_len), dtype=np.uint8))

    def get_tile_mask(tile_id):
        # Return mask by id wherever it is
        if tile_id in saltdata.id2mask:
            return saltdata.id2mask[tile_id]
        elif tile_id in saltdata.id2pred_cor:
            return saltdata.id2pred_cor[tile_id]
        elif tile_id in saltdata.id2pred:
            return saltdata.id2pred[tile_id]

    def process_mosaic(mosaic):
        # Process mosaics with 3 tiles or more
        if len(mosaic) > 2:
            # Find the terminal level
            is_term_mat = np.zeros_like(mosaic.array, dtype=np.bool)
            for i in range(mosaic.array.shape[0]):
                for j in range(mosaic.array.shape[1]):
                    tile_id = mosaic.array[i,j]
                    if tile_id is not None:
                        tile_id = tile_id
                        tile_mask = get_tile_mask(tile_id)
                        is_term_mat[i, j] = is_terminal(tile_mask)
            term_level = 0
            for i in range(mosaic.array.shape[0]):
                h_strip = mosaic.array[i,:][np.newaxis, :]
                is_term = [False] * h_strip.shape[1]
                for j, tile in enumerate(h_strip[0]):
                    if tile is not None:
                        tile_id = tile
                        if tile_id in saltdata.id2mask:
                            is_term[j] = (is_term_mat[i, j]\
                                          and np.count_nonzero(saltdata.id2mask[tile_id]) > (img_side_len**2)/20)
                        elif tile_id in saltdata.id2pred:
                            is_term[j] = (is_term_mat[i, j]\
                                          and np.count_nonzero(saltdata.id2pred[tile_id]) > (img_side_len**2)/20)
                if any(is_term):
                    term_level = i
                    break
            # Fix vertical columns of tiles
            for i in range(mosaic.array.shape[1]):
                strip = mosaic.array[:,i][:, np.newaxis]
                process_strip(strip, term_level)

            # Add missed corners
            for i in range(mosaic.array.shape[0]):
                for j in range(mosaic.array.shape[1]):
                    if True:#is_term_mat[i, j]:
                        if mosaic.array[i,j] is not None:
                            # Left
                            if i+1 < mosaic.array.shape[0] and j+1 < mosaic.array.shape[1]:
                                
                                if mosaic.array[i+1, j+1] is not None:
                                    l_id = mosaic.array[i, j+1]
                                    if (l_id in saltdata.id2pred\
                                            and np.count_nonzero(get_tile_mask(l_id)) < img_side_len**2/1000)\
                                            or l_id in saltdata.id2pred_cor:

                                        tile1 = get_tile_mask(mosaic.array[i,j])
                                        tile2 = get_tile_mask(mosaic.array[i+1, j+1])

                                        fix = find_fit_corners(tile2, tile1, True)
                                        if fix is not None:
                                            if l_id not in saltdata.id2pred_cor:
                                                saltdata.add_pred_cor(l_id, fix)
                                            else:
                                                saltdata.id2pred_cor[l_id] = saltdata.id2pred_cor[l_id] + fix
                            # Right
                            if i+1 < mosaic.array.shape[0] and j-1 > 0:
                                if mosaic.array[i+1, j-1] is not None:
                                    if mosaic.array[i+1, j] is not None:
                                        r_id = mosaic.array[i, j-1]
                                        if (r_id in saltdata.id2pred\
                                                and np.count_nonzero(get_tile_mask(r_id)) < img_side_len**2/1000)\
                                                or r_id in saltdata.id2pred_cor:
                                            tile1 = get_tile_mask(mosaic.array[i,j])
                                            tile2 = get_tile_mask(mosaic.array[i+1, j-1])

                                            fix = find_fit_corners(tile2, tile1, False)
                                            if fix is not None:
                                                if r_id not in saltdata.id2pred_cor:
                                                    saltdata.add_pred_cor(r_id, fix)
                                                else:
                                                    saltdata.id2pred_cor[r_id] = saltdata.id2pred_cor[mosaic.array[i, j-1]] + fix

            for i in range(mosaic.array.shape[1]):
                strip = mosaic.array[:,i][:, np.newaxis]
                zero_under_salt(strip, term_level)

    # Fill holes in images
    for f_name in os.listdir(mean_prediction_dir):
        mask = cv2.imread(join(mean_prediction_dir, f_name),
                          cv2.IMREAD_GRAYSCALE)
        mask = fix_holes(mask)
        cv2.imwrite(join(postprocessed_prediction_dir, f_name), mask)

    # Postprocess mosaics
    saltdata = SaltData(mosaic_csv_path=MOSAIC_PATH,
                        pred_dir=postprocessed_prediction_dir)
    for mosaic_id in saltdata.mosaics.mosaic_id2mosaic:
        process_mosaic(saltdata.mosaics.mosaic_id2mosaic[mosaic_id])

    # Save corrected tiles
    for f_name in os.listdir(postprocessed_prediction_dir):
        f_id = filename_without_ext(f_name)
        if f_id in saltdata.id2pred_cor:
            cv2.imwrite(join(postprocessed_prediction_dir, f_name),
                        saltdata.id2pred_cor[f_id])
