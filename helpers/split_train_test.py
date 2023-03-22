from doctest import testfile
import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

def split_train_test_by_mtdata(DATA_PATH, MDT_PATH, speakers = None, random_state = 0):
  '''
  split data into train and test and further divide audio based on target into folders.
  This split is done based on metadata file . 
  '''

  sdr_df = pd.read_csv(MDT_PATH, sep='\t', header=0, index_col='Unnamed: 0')

  # create directory for split data
  TRAIN_DIR, TEST_DIR = None, None
  PARENT_DIR = '/'.join(DATA_PATH.split('/')[:-1])
  DATA_DIR   = PARENT_DIR + '/data'
  TRAIN_DIR  = DATA_DIR   + '/train'
  TEST_DIR   = DATA_DIR   + '/test'
  DEV_DIR    = DATA_DIR   + '/dev'

  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
  else:
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    
  if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

  if not os.path.exists(TEST_DIR):  
    os.makedirs(TEST_DIR)
  
  if not os.path.exists(DEV_DIR):  
    os.makedirs(DEV_DIR)


  for idx, row in sdr_df.iterrows():
    SPLIT_TYP, filename = row['split'], row['file']
    filename            = filename.split('/')[1]
    speakername         = str(filename.split('_')[1])
       
    if SPLIT_TYP == 'TRAIN':
      if speakers == None or speakername in speakers:
        shutil.copy( os.path.join(DATA_PATH, filename) , os.path.join(TRAIN_DIR, filename))
    elif SPLIT_TYP == 'DEV':
      shutil.copy( os.path.join(DATA_PATH, filename) , os.path.join(DEV_DIR, filename))
    elif SPLIT_TYP == 'TEST':
      shutil.copy( os.path.join(DATA_PATH, filename) , os.path.join(TEST_DIR, filename))
    
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
    shutil.copy( os.path.join(ABSOLUTE_PTH, train_file) , os.path.join(TRAIN_DIR, train_file))

  for test_file in test_files:
    shutil.copy( os.path.join(ABSOLUTE_PTH, test_file) , os.path.join(TEST_DIR, test_file))


def split_train_test_speaker_by_mtdata(DATA_PATH, MDT_PATH, random_state = 0):
  '''
  split data into train and test for SPECIFIC SPEAKERS. Further divide audio based on target into folders.
  This split is done based on metadata file . 
  '''

  sdr_df = pd.read_csv(MDT_PATH, sep='\t', header=0, index_col='Unnamed: 0')

  for speaker in list(set(sdr_df.speaker.values)):
    # create directory for split data
    TRAIN_DIR, TEST_DIR = None, None
    PARENT_DIR = '/'.join(DATA_PATH.split('/')[:-1])
    DATA_DIR   = PARENT_DIR + '/data' + f'/{speaker}'
    TRAIN_DIR  = DATA_DIR   + '/train'
    TEST_DIR   = DATA_DIR   + '/test'
    DEV_DIR    = DATA_DIR   + '/dev'

    if not os.path.exists(DATA_DIR):
      os.makedirs(DATA_DIR)
    else:
      shutil.rmtree(DATA_DIR)
      os.makedirs(DATA_DIR)
      
    if not os.path.exists(TRAIN_DIR):
      os.makedirs(TRAIN_DIR)

    if not os.path.exists(TEST_DIR):  
      os.makedirs(TEST_DIR)
    
    if not os.path.exists(DEV_DIR):  
      os.makedirs(DEV_DIR)

    sdr_speaker_df = sdr_df.loc[sdr_df['speaker'] == speaker,['split','file']]

    for idx, row in sdr_speaker_df.iterrows():
      SPLIT_TYP, filename = row['split'], row['file']
      filename            = filename.split('/')[1]

      if SPLIT_TYP == 'TRAIN':
        shutil.copy( os.path.join(DATA_PATH, filename) , os.path.join(TRAIN_DIR, filename))
      elif SPLIT_TYP == 'DEV':
        shutil.copy( os.path.join(DATA_PATH, filename) , os.path.join(DEV_DIR, filename))
      elif SPLIT_TYP == 'TEST':
        shutil.copy( os.path.join(DATA_PATH, filename) , os.path.join(TEST_DIR, filename))



TEST_PTH  = os.path.join(os.getcwd(), 'data', 'test')
DATA_PATH = os.path.join(os.getcwd(), 'speech_data')
MDT_PATH  = os.path.join(os.getcwd(), 'SDR_metadata.tsv')

split_train_test_speaker_by_mtdata(DATA_PATH, MDT_PATH)