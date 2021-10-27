from tensorboard.backend.event_processing import event_accumulator


def get_value_from_event(event_file, keys=(), verbose=True):
  if isinstance(keys, (str, )):
    keys = [keys, ]

  ea = event_accumulator.EventAccumulator(event_file)
  ea.Reload()
  if verbose: print(ea.scalars.Keys())

  loaded_data = {}
  for key in keys:
    data = ea.scalars.Items(key)
    data_list = [(i.step, i.value) for i in data]
    loaded_data[key] = data_list

  return loaded_data
