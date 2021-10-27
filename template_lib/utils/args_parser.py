import argparse


def none_or_str(value):
  if value.lower() == 'none':
    return None
  return value


def true_or_false(value):
  if value.lower() == 'true':
    return True
  return False


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=none_or_str, default='')
  parser.add_argument('--command', type=none_or_str, default='')
  parser.add_argument('--outdir', type=str, default='results/temp')
  parser.add_argument('--overwrite_opts', type=true_or_false, default=False)

  parser.add_argument('--world_size', type=int, default=1)
  parser.add_argument('--resume', type=true_or_false, default=False)
  parser.add_argument('--resume_path', type=none_or_str, default='')
  parser.add_argument('--resume_root', type=none_or_str, default='')
  parser.add_argument('--evaluate', type=true_or_false, default=False)
  parser.add_argument('--evaluate_path', type=none_or_str, default='')
  parser.add_argument('--finetune', type=true_or_false, default=False)
  parser.add_argument('--finetune_path', type=none_or_str, default='')
  return parser
