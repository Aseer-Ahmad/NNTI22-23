from doctest import testfile
import os
import shutil

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

def split_train_test_random(ABSOLUTE_PTH, random_state = 0):
  '''
  split data into train and test 
  and further divide audio based on target into folders.
  This split is done randomly. 
  '''
  pass

def split_train_test_irregular(ABSOLUTE_PTH, random_state = 0 ):
  pass

def split_train_test_by_user(ABSOLUTE_PTH, test_size = 0.2, random_state = 0):
  '''
  split data into train and test and further divide audio based on target into folders.
  This split is done based on user hence a stratified split . The data is then stored in a 
  separate directory and is created anew everytime. 
  '''

  train_files, test_files = None, None
  df          = pd.DataFrame(columns = ['file', 'user_target'])
  audio_files = os.listdir(ABSOLUTE_PTH)
  sss         = StratifiedShuffleSplit(n_splits = 1, test_size=test_size, random_state=random_state)

  # create directory for split data
  TRAIN_DIR, TEST_DIR = None, None
  PARENT_DIR = '/'.join(ABSOLUTE_PTH.split('/')[:-1])
  DATA_DIR   = PARENT_DIR + '/data'
  TRAIN_DIR = DATA_DIR + '/train'
  TEST_DIR  = DATA_DIR + '/test'

  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
  else:
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    
  if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

  if not os.path.exists(TEST_DIR):  
    os.makedirs(TEST_DIR)


  # create target wise directory
  for i in range(10):
    TARGET_TRAIN_DIR = os.path.join(TRAIN_DIR, str(i))
    TARGET_TEST_DIR = os.path.join(TEST_DIR, str(i))
    os.makedirs(TARGET_TRAIN_DIR)
    os.makedirs(TARGET_TEST_DIR)

    
  # create data frame for stratified split
  for audio_file in audio_files:
    parse  = audio_file.split('_') 
    user   = parse[1]
    target = parse[0]
    df = df.append({'file':audio_file, 'user_target':str(user+target)}, ignore_index = True)
    
  X, Y  = df['file'], df['user_target']

  # fetch files from train test split
  for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
    train_files = X[train_index]
    test_files  = X[test_index]

  for train_file in train_files:
    target = train_file.split('_')[0]
    shutil.copy( os.path.join(ABSOLUTE_PTH, train_file) , os.path.join(TRAIN_DIR, target, train_file))

  for test_file in test_files:
    target = test_file.split('_')[0]
    shutil.copy( os.path.join(ABSOLUTE_PTH, test_file) , os.path.join(TEST_DIR, target, test_file))


split_train_test_by_user('/home/cepheus/My GIT/NNTI 22-23/speech_data', test_size=0.1)
