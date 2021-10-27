from collections import OrderedDict
from pprint import pformat

from template_lib.utils import seed_utils

from . import exe_dict, trainer_dict


def train(args, myargs):
  myargs.config = getattr(myargs.config, args.command)
  config = myargs.config
  print(pformat(OrderedDict(config)))
  trainer = trainer_dict[args.command](args=args, myargs=myargs)

  seed_utils.set_random_seed(config.seed)

  if args.evaluate:
    trainer.evaluate()
    return

  if args.resume:
    trainer.resume()
  elif args.finetune:
    trainer.finetune()

  # Load dataset
  trainer.dataset_load()

  trainer.train()


def main(args, myargs):
  exec('%s(args, myargs)'%exe_dict[args.command])