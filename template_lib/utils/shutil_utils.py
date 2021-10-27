import os


def ignoreAbsPath(path):
  """
  For shutil.copytree func
  :param path:
  :return:
  """
  path = [os.path.abspath(elem) for elem in path]

  def ignoref(directory, contents):
    ig = [f for f in contents if
          os.path.abspath(os.path.join(directory, f)) in path]
    return ig

  return ignoref


def ignoreNamePath(path):
  """
  For shutil.copytree func
  :param path:
  :return:
  """
  path += ['.idea', '.git', '.pyc']
  def ignoref(directory, contents):
    ig = [f for f in contents if
          (any([f.endswith(elem) for elem in path]))]
    return ig

  return ignoref
