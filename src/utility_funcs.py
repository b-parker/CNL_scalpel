import numpy as np
import functools
import nibabel as nb
import subprocess as sp
import shlex

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

