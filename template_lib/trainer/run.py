from . import trainer_dict


def trainer_create(args, myargs):
  config = myargs.config.trainer
  myargs.logger.info('Create trainer: %s', config.type)
  trainer = trainer_dict[config.type](args=args, myargs=myargs)
  return trainer


def main(args, myargs):
  trainer = trainer_create(args, myargs)
  from template_lib.trainer import train
  train.main(trainer=trainer, args=args, myargs=myargs)