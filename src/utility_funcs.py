import numpy as np
import functools
import nibabel as nb
import subprocess as sp
import shlex
import os
import pandas as pd

def memoize(obj):
  cache = obj.cache = {}

  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args) + str(kwargs)
    if key not in cache:
      cache[key] = obj(*args, **kwargs)

    return cache[key]
  return memoizer

def mris_convert_command(filepath, custom_filename=None):
  """
  Run freesurfer mri_convert on surface
  """
  if custom_filename == None:
    cmd = f'mris_convert {filepath} {filepath}.gii'
  else:
    cmd = f'mris_convert {filepath} {filepath}.{custom_filename}.gii'
  print(f'Executing mris_convert for {filepath}')
  args = shlex.split(cmd)
  run_command = sp.Popen(args)


def get_unique_labels(subjects_dir: str, dataset: str):
  """
  Search through subjects dir and get all labels / counts as DataFrame

  INPUT:
  ______
  subjects_dir: str - filepath to subject directory
  dataset: str - name of dataset for dataframe

  OUTPUT:
  ______
  labels_df: pd.DataFrame - DataFrame of labels, columns = ['dataset', 'label', 'count']
  

  """

  subjects = os.listdir(subjects_dir)
  labels_df = pd.DataFrame(columns=['dataset', 'label', 'count'])
  for subject in subjects:
    label_dir = os.path.join(subject, 'label')
    if os.path.exists(label_dir):
      labels = os.listdir(label_dir)
      for label in labels:
        if label in labels_df['label'].to_list():
          labels_df[labels_df['label'] == label]['count'] += 1
        else:
          new_row = [dataset, label, 1]
          labels_df.loc[len(labels_df)] = new_row
  
  return labels_df