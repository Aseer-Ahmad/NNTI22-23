import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def split_train_test(PATH, random_state = 0):
  '''
  split data into train and test 
  and further divide audio based on target into folders.
  This split is done randomly. 
  '''


def split_train_test_by_user(PATH, random_state = 0):
  '''
  split data into train and test 
  and further divide audio based on target into folders.
  This split is done based on user hence a stratified split . 
  '''
  df          = pd.DataFrame(columns = ['file', 'user', 'target', 'user_target'])
  audio_files = os.listdir(PATH)
  
  for audio_file in audio_files:
    parse  = audio_file.split('_') 
    user   = parse[1]
    target = parse[0]
    df = df.append({'file':audio_file, 'user':user, 'target':target ,'user_target':str(user+target)}, ignore_index = True)
    

    

  




