import os
import cv2
import time
import tqdm
import json
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

img_train_dir = '../../data/train/images/'
img_test_dir = '../../data/test/images/'
images_names = ['14b50d2c26', '31f29ada31', '26c403ce6d', '3eef3e190f', 'b2d4c41f68', '14c2834524', 'cb36e2e2ae', '971ff26c0d', 'a0891f7e3e',
'0b9874fd4f', '476bde9dbf', 'ad068f08d1', '4d33311a1e', '7dfdf6eeb8', '27838f7f46', 'f27a3ca60e', 'da324cbdde', 'd0bdfc3217', '393048bc6c',
'eb84dfdf18', 'c100d07aa9', '3d0f8c81ce', 'a667ba911e', '973a14e68f', 'cf5be83765', 'b174e802d6', '3768877c25', '1d6f51de05', 'c24e12d0cf',
'f40bfdeb45', '55f1ea73b3', 'f91af813d6', '8d0c065461', 'f96f608c5a', '4b366ad75d', 'de15d35ebc', 'e5c26c8634',
'f8ba90c991', '59dabe134f', '5f723327ef', 'd9d7c77e13', 'b0beff63b4', '4aa5b85fb1', '4b6bc93e6c', 'cb6c6c15ce', '595deae139', '5420263191',
'1986fcadda', '0c14b0fe28', 'd639f4c3ae', '499237cf6c', 'dc71cdae70', '9472731dfd', '6230e29f51', '34f944baa4', '3ab326340b',
'7cf980df36', '2fecaf1c54', 'c2ec2c9de4', '8a58fe70f8', 'd3e4630469', '9a91290888', 'c5d6528a35', '3be0c0be8e', 'fca4d42846', '70b7e1c459',
'abd9ba8120', 'cbc5bd4021', '52ae147d60', '8a6090c9ec', '7205efa791', '70afc514d2', '4ca0fd980f', '668cf61ce0', 'f448297676', 'd96f1003cf',
'c4f2855630', '44d28db2dc', '94500b7464', 'ea39579316', '753ac9f3ed', 'a0e6d8b0a7', '7db7a5fb8f', '8299b97f38', '73d3d08aff', '095473ab35',
'61b6c452ad', '6d69267940', 'f9f6588f79', '54629440c9', 'cd90c05864', '0bbaa6d56a', '5299cd2c33', 'f069247787',
]
THRES = 5
diff_order = 1
# Outputs
mosaic_dict_path = '../../data/mosaic/mosaic1.json'
clusters_dict_path = '../../data/mosaic/clusters1.json'
img_mosaics_path = '../../data/mosaic/imgs2/'
mean_hist_path = '../../data/mosaic/mean_hist.npy'
save_mtrx = True
left_mtrx_path = '../../data/mosaic/left1.npy'
top_mtrx_path = '../../data/mosaic/top1.npy'

images_names = os.listdir(img_train_dir)
#images_names += os.listdir(img_test_dir)

mean_hist = np.load(mean_hist_path)
mean_hist[0] = mean_hist[-1]

def make_descriptor(img):
    edges = [img[:, 0], img[:, -1], img[0, :], img[-1, :]]
    return [(np.diff(edge.astype(np.int), n=diff_order), edge)
            for edge in edges]

def metric(desc1, desc2):
    return np.mean((desc1[0]-desc2[0])**2) + np.mean((desc1[1]-desc2[1])**2)

def create_mosaic_dict(images_paths, tiles, tile_to_cluster):
    tiles_dicts = {}
    for i in range(n_samples):
        tiles_dicts[i] = {
            'path': images_paths[i],
            'cluster': tile_to_cluster[tiles[i][0]],
            'neighbours': tiles[i][1:]
        }
    return tiles_dicts

def hist_match(source, hist):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches a given one

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template histogram
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values = np.linspace(0, 255, 256).astype(np.int)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(hist).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

n_samples = len(images_names)

images_paths = []
for img_name in images_names:
    if img_name[-3:] != 'png':
        train_path = os.path.join(img_train_dir, img_name+'.png')
        test_path = os.path.join(img_test_dir, img_name+'.png')
    else:
        train_path = os.path.join(img_train_dir, img_name)
        test_path = os.path.join(img_test_dir, img_name)
    if os.path.exists(train_path):
        images_paths.append(train_path)
    elif os.path.exists(test_path):
        images_paths.append(test_path)

# Check if everything is ok
print("Samples:", n_samples, "Ok:", len(images_paths)==len(images_names))

imgs = [hist_match(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), mean_hist)
        for img_path in images_paths]
#imgs = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #for img_path in images_paths]



c_time = time.time()
descriptors = [make_descriptor(img) for img in imgs]

def calc_metrics(ij):
    i, j = ij
    if i != j:
        left_metr = metric(descriptors[i][0], descriptors[j][1])
        top_metr = metric(descriptors[i][2], descriptors[j][3])
    else:
        left_metr = top_metr = 10000 # Just a big number
    return left_metr, top_metr

left_matrix = np.zeros((n_samples, n_samples))
top_matrix = np.zeros((n_samples, n_samples))
print("Descriptors matrix creation")
with Pool(processes=12) as pool:
    for i in tqdm.tqdm(range(len(descriptors))):
        metr = pool.imap(calc_metrics, [(i, j) for j in
                                        range(len(descriptors))])
        for j, m in enumerate(metr):
            left_matrix[i, j] = m[0]
            top_matrix[i, j] = m[1]

if save_mtrx:
    np.save(left_mtrx_path, left_matrix)
    np.save(top_mtrx_path, top_matrix)

tiles = [[i, None, None, None, None] for i  in range(n_samples)]

print("Create left neighbours")
left_thres = np.median(left_matrix, axis=1)/THRES
left_idx = np.argsort(left_matrix, axis=1)[:,0]

for i in range(n_samples):
    j = int(left_idx[i])
    if left_matrix[i, j] < left_thres[i]:
        tiles[i][1] = j
        tiles[j][3] = i

print("Create top neighbours")
top_thres = np.median(top_matrix, axis=1)/THRES
top_idx = np.argsort(top_matrix, axis=1)[:,0]

for i in range(n_samples):
    j = int(top_idx[i])
    if top_matrix[i, j] < top_thres[i]:
        tiles[i][2] = j
        tiles[j][4] = i


clusters = []
n_clusters = 1
tile_to_cluster = defaultdict(int)
print("Clusters assigning")
for tile in tqdm.tqdm(tiles):
    tile_cluster = tile_to_cluster[tile[0]]
    if tile_cluster == 0:
        tile_to_cluster[tile[0]] = n_clusters
        n_clusters += 1
    tile_cluster = tile_to_cluster[tile[0]]
    for idx in tile[1:]:
        if idx is not None:
            if tile_to_cluster[idx] == 0:
                tile_to_cluster[idx] = tile_cluster
            else:
                # TODO Think about collisions solution
                pass

print(time.time()-c_time, "sec passed")

tiles_dicts = create_mosaic_dict(images_paths, tiles, tile_to_cluster)
with open(mosaic_dict_path, 'w') as fout:
    json.dump(tiles_dicts, fout, indent=2)


img_size = (101,101)
cls_id = 3

def coords_for_open_list(i, j, neighbours, open_list, close_list):
    to_open_list = []
    for n, k in enumerate(neighbours):
        if k is not None and k not in close_list:
            close_list.append(k)
            i_k = i
            j_k = j
            if n == 0:
                i_k -= 1  # left
            elif n == 1:
                j_k -= 1  # top
            elif n == 2:
                i_k += 1  # right
            else:
                j_k += 1  # bottom
            to_open_list.append((i_k, j_k, k))
    return open_list+to_open_list, close_list

print("Assemble mosaics")
def create_mosaics(tiles_dicts):

    def coords_ij(ij, lim):
        i, j = ij
        i_max, j_max, i_min, j_min = lim
        x_min = (i-i_min)*img_size[1]
        x_max = (i+1-i_min)*img_size[1]
        y_min = (j-j_min)*img_size[0]
        y_max = (j+1-j_min)*img_size[0]
        return x_min, x_max, y_min, y_max
    
    def add_img(mosaic, ij, lim, el):
        i, j = ij
        i_max, j_max, i_min, j_min = lim
        tile = tiles_dicts[el[2]]
        if el[1] < j_min:
            mosaic = np.pad(mosaic, ((img_size[0], 0), (0, 0)), mode='constant')
            j_min -= 1
        if el[1] > j_max:
            mosaic = np.pad(mosaic, ((0, img_size[0]), (0, 0)), mode='constant')
            j_max += 1
        if el[0] < i_min:
            mosaic = np.pad(mosaic, ((0, 0), (img_size[0], 0)), mode='constant')
            i_min -= 1
        if el[0] > i_max:
            mosaic = np.pad(mosaic, ((0, 0), (0, img_size[0])), mode='constant')
            i_max += 1
        #print('res mosaic', mosaic.shape)
        x_min, x_max, y_min, y_max = coords_ij(el[:2],
                                     (i_max, j_max, i_min, j_min))
        #print("Add", el[:2], el[2], [x_min, x_max, y_min, y_max])
        mosaic[y_min:y_max, x_min:x_max] = hist_match(cv2.imread(tile['path'],
                       cv2.IMREAD_GRAYSCALE), mean_hist)
        return mosaic, i_max, j_max, i_min, j_min
        
        
    mosaics = []
    dicts_to_save = []
    while len(tiles_dicts) > 0:
        min_idx = min(tiles_dicts.keys())
        tile = tiles_dicts[min_idx]
        del tiles_dicts[min_idx]
        mosaics.append(cv2.imread(tile['path'],
                       cv2.IMREAD_GRAYSCALE))
        i = 1
        j = 1
        i_max = 1
        j_max = 1
        i_min = 1
        j_min = 1
        open_list = []
        close_list = [min_idx]
        open_list, close_list = coords_for_open_list(i, j, tile['neighbours'],
                                                     open_list, close_list)
        dicts_to_save.append([])
        dicts_to_save[-1].append({'i': i, 'j': j, 'path': tile['path']})
        while len(open_list) > 0:
            el = open_list.pop(0)
            if el[2] in tiles_dicts.keys():
                mosaics[-1], i_max, j_max, i_min, j_min = add_img(mosaics[-1], (i, j), (i_max, j_max, i_min, j_min), el)
                i,j = el[:2]
                open_list, close_list = coords_for_open_list(i, j, tiles_dicts[el[2]]['neighbours'],
                                                  open_list, close_list)
                dicts_to_save[-1].append({'i': i, 'j': j, 'path': tiles_dicts[el[2]]['path']})
                del tiles_dicts[el[2]]
                #print(open_list, close_list)
    return mosaics, dicts_to_save

mosaics, dicts_to_save = create_mosaics(tiles_dicts)

for i, mosaic in enumerate(mosaics):
    cv2.imwrite(os.path.join(img_mosaics_path, str(i)+'.png'),
                mosaic)

with open(clusters_dict_path, 'w') as fout:
    json.dump(dicts_to_save, fout, indent=2)
