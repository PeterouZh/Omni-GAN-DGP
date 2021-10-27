import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)


if __name__ == '__main__':
  args = parser.parse_args()
  with open(os.path.expanduser(args.input)) as f:
    jf = json.load(f)
  with open(os.path.expanduser(args.output), 'w') as f:
    json.dump(jf, f)
