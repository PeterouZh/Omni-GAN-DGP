import os
import shutil
import argparse


parser = argparse.ArgumentParser(description='copy tool')
parser.add_argument('-s', type=str)
parser.add_argument('-d', type=str)
parser.add_argument('-l', nargs="+")

cfg = parser.parse_args()


def main():
  try:
    import moxing as mox
  except:
    import traceback
    traceback.print_exc()

  print('=> Copy file(s) from %s to %s ...' % (cfg.s, cfg.d))
  assert mox.file.is_directory(cfg.s)
  for file in cfg.l:
    src = f"{cfg.s}/{file}"
    dst = f"{cfg.d}/{file}"
    print(f'   Copying file(s) from {src} to {dst} ...')
    if mox.file.is_directory(src):
      mox.file.copy_parallel(src, dst)
    else:
      mox.file.copy(src, dst)
  print('=> End copy file(s) from %s to %s ...' % (cfg.s, cfg.d))


if __name__ == '__main__':
  """
  python template_lib/modelarts/scripts/copy_file_list.py \
          -s s3://bucket-7001/ZhouPeng/results/Omni-GAN-ImageNet/OmniInrGAN_ImageNet256/train_ImageNet256-20210214_134020_858 \
          -d /cache/Omni-GAN-ImageNet/results/OmniInrGAN_ImageNet256/train_ImageNet256-20210214_134020_858 \
          -l  biggan/weights \
              biggan/logs \
              config_command.yaml \
              config.yaml \
              log.txt \
              textdir/
  """
  main()
