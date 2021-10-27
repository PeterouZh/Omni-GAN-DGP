import pprint
import argparse
import sys
sys.path.insert(0, '.')
import tempfile

from template_lib.utils import get_filelist_recursive
from template_lib.v2.logger.logger import get_file_logger


def main():

  file_list = get_filelist_recursive(directory=args.source_dir, ext=args.ext, sort=True)
  print(f"number of items: {len(file_list)}")

  if not args.outfile:
    fd, path = tempfile.mkstemp()
    args.outfile = path

  out_f = get_file_logger(args.outfile, stream=True)
  for path in file_list:
    out_f.info_msg(path)
  print(f"outfile: {args.outfile}")
  pass

if __name__ == '__main__':
  """
  python3 template_lib/proj/ubuntu/get_data_list.py \
    --source_dir  --outfile  --ext *.png
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str, default="")
  parser.add_argument('--outfile', type=str, default="")
  parser.add_argument('--ext', type=str, nargs='+', default=["*.png"])

  args = parser.parse_args()
  pprint.pprint(vars(args))
  main()

