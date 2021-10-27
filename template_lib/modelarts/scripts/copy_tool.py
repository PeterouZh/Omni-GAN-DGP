import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='copy tool')
parser.add_argument('-s', type=str)
parser.add_argument('-d', type=str)
parser.add_argument('-t', type=str, default="copy", choices=['copytree', 'copy',
                                                             'copytree_nooverwrite', 'copy_nooverwrite'])
parser.add_argument('-b', type=str, default=None)

cfg = parser.parse_args()


def main():
  try:
    import moxing as mox
  except:
    import traceback
    traceback.print_exc()

  if cfg.b is not None:
    from tempfile import TemporaryDirectory
    from template_lib.utils import make_zip
    with TemporaryDirectory() as dirname:
      print(f'=> Backup dir {cfg.s} to \n {dirname}')
      mox.file.copy_parallel(cfg.s, dirname)
      print(f'=> Backup {dirname} to \n {cfg.b}')
      os.makedirs(os.path.dirname(cfg.b), exist_ok=True)
      make_zip(source_dir=dirname, output_filename=cfg.b)

  print('=> Copy file(s) from %s to %s ...' % (cfg.s, cfg.d))
  if cfg.t == "copytree":
    mox.file.copy_parallel(cfg.s, cfg.d)
  elif cfg.t == 'copy':
    mox.file.copy(cfg.s, cfg.d)
  elif cfg.t == 'copytree_nooverwrite':
    if os.path.exists(cfg.d):
      print('Skip copying, %s exist!' % cfg.d)
      return
    mox.file.copy_parallel(cfg.s, cfg.d)
  elif cfg.t == 'copy_nooverwrite':
    if os.path.exists(cfg.d):
      print('Skip copying, %s exist!' % cfg.d)
      return
    mox.file.copy_parallel(cfg.s, cfg.d)
  else:
    assert 0
  print('=> End copy file(s) from %s to %s ...' % (cfg.s, cfg.d))


if __name__ == '__main__':
  main()
