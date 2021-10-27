

def main(trainer, args, myargs):
  config = myargs.config

  from template_lib.utils import seed_utils
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



