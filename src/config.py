from os.path import join

DATA_DIR = '/workdir/data/'
TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
DEPTHS_PATH = join(DATA_DIR, 'depths.csv')
TRAIN_FOLDS_PATH = join(DATA_DIR, 'train_folds.csv')
TRAIN_CSV_PATH = join(DATA_DIR, 'train.csv')
SAMPLE_SUBM_PATH = join(DATA_DIR, 'sample_submission.csv')
MEAN_HIST_PATH = join(DATA_DIR, 'mean_hist.npy')
N_FOLDS = 6
