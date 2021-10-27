# import tensorflow.keras as keras
import sys
from collections import OrderedDict
import datetime
import warnings
import numpy as np
import glob
import keras
import logging
import os, shutil
import re
import tensorflow as tf

from . import utils


class TensorboardOnBatch(object):

  def __init__(self, log_dir, model, histogram_freq, batch_size=32,
               write_graph=False, write_grads=False, write_images=False, del_old_logs=False):
    """# This example shows how to use keras TensorBoard callback with model.train_on_batch

    :return:
    """
    logging.info("Tensorboard save to %s", log_dir)
    if os.path.exists(log_dir) and del_old_logs:
      logging.info("Delete old tensorboard event: %s", log_dir)
      shutil.rmtree(log_dir)

    self.model = model
    # Create the TensorBoard callback, which we will drive manually
    self.tensorboard = keras.callbacks.TensorBoard(
      log_dir=log_dir,
      histogram_freq=histogram_freq,
      batch_size=batch_size,
      write_graph=write_graph,
      write_grads=write_grads,
      write_images=write_images
    )
    self.tensorboard.set_model(model)

    self.summary_placeholders = {}
    self.summary_ops = {}
    self.sess = keras.backend.get_session()

  def log_args(self, args_od, args):
    import json
    args = OrderedDict(sorted(vars(args).items()))
    for arg in args_od:
      if arg in args:
        args.__delitem__(arg)

    args_od_str = json.dumps(args_od, indent=2)
    # for pretty indent
    args_od_str = args_od_str.replace('\n', '  \n>')
    self.log_text(step=0, names='args_od', values=args_od_str, curent_time=True)

    args_str = str(dict(args))
    args_str = args_str.replace(' ', '  \n>')
    self.log_text(step=0, names='args', values=args_str, curent_time=True)
    pass

  def log_epoch(self, epoch):
    step = epoch
    name = 'epoch'
    value = 'epoch: %d' % epoch
    self.log_text(step=step, names=name, values=value, curent_time=True)

  def _named_logs(self, logs):
    """# Transform train_on_batch return value
    # to dict expected by on_batch_end callback

    :param logs: list
    :return:
    """
    result = {}
    for l in zip(self.model.metrics_names, logs):
      result[l[0]] = l[1]
    return result

  def on_epoch_end(self, step, logs):
    logs_dict = self._named_logs(logs)
    self.tensorboard.on_epoch_end(step, logs_dict)

  def on_train_end(self):
    self.tensorboard.on_train_end(None)

  def log_value(self, step, names, values):
    if not isinstance(names, (list, tuple)):
      names = [names, ]
    if not isinstance(values, (list, tuple)):
      values = [values, ]
    for name, value in zip(names, values):
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.tag = name
      summary_value.simple_value = value
      self.tensorboard.writer.add_summary(summary, step)
      self.tensorboard.writer.flush()

  def log_renamed_value(self, step, old_names, new_names, logs):
    """ rename value's tag for showing different value in the same figure in tensorboard

    :param step:
    :param old_names:
    :param new_names:
    :param logs:
    :return:
    """
    assert len(old_names) == len(new_names)
    logs_dict = self._named_logs(logs)
    names = []
    values = []
    for old_name, new_name in zip(old_names, new_names):
      if old_name in logs_dict:
        names.append(new_name)
        values.append(logs_dict[old_name])
    self.log_value(step, names=names, values=values)

  def log_images(self, step, names, values, max_images=6):
    """

    :param step:
    :param names:
    :param values: image range [0, 255], uint8
    :param max_images:
    :return:
    """
    if not isinstance(names, (list, tuple)):
      names = [names, ]
    if not isinstance(values, (list, tuple)):
      values = [values, ]
    for name, value in zip(names, values):
      if name not in self.summary_ops:
        self.summary_placeholders[name] = tf.placeholder('uint8', [None] + list(value.shape[1:]),
                                                         name='{0}_pl'.format(name))
        self.summary_ops[name] = tf.summary.image(name='{0}_image'.format(name),
                                                  tensor=self.summary_placeholders[name],
                                                  max_outputs=max_images)
      value = np.clip(value, 0, 255).astype(np.uint8)
      images = self.sess.run(self.summary_ops[name], {self.summary_placeholders[name]: value})
      self.tensorboard.writer.add_summary(images, step)
      self.tensorboard.writer.flush()

  def log_merged_images(self, step, names, values):
    merged_images = []
    for value in values:
      value = utils.BatchImages.merge_batch_images_to_one_auto(b_images=value, keep_dim=True)
      merged_images.append(value)
    self.log_images(step=step, names=names, values=merged_images)

  def log_text(self, step, names, values, curent_time=False):
    if not isinstance(names, (list, tuple)):
      names = [names, ]
    if not isinstance(values, (list, tuple)):
      values = [values, ]
    for name, value in zip(names, values):
      name = name + '_text'
      if name not in self.summary_ops:
        self.summary_placeholders[name] = tf.placeholder(tf.string,
                                                         name='{0}_pl'.format(name))
        self.summary_ops[name] = tf.summary.text('{0}'.format(name),
                                                 self.summary_placeholders[name])

      if curent_time:
        time_str = str((datetime.datetime.utcnow() + datetime.timedelta(hours=8))
                       .strftime('%y-%m-%d %H:%M:%S'))
        value = str(value) + '  [%s]' % time_str

      text = self.sess.run(self.summary_ops[name], {self.summary_placeholders[name]: value})
      self.tensorboard.writer.add_summary(text, step)
      self.tensorboard.writer.flush()


class ModelCheckpointOnBatch(keras.callbacks.ModelCheckpoint):
  """filepath = os.path.join(filepath, 'epoch-{epoch:05d}-psnr-{' + monitor + ':.4f}.h5')"""

  def __init__(self, model, saved_path, filename, monitor='val_loss', verbose=0,
               save_best_only=False, save_weights_only=False,
               mode='auto', period=1, after_epoch=1, max_to_keep=10):

    assert mode in ['max', 'min']
    if not os.path.exists(saved_path):
      os.makedirs(saved_path)
    # TODO: delete useless model

    filepath = os.path.join(saved_path, filename)
    super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    self.saved_path = saved_path
    self.mode = mode
    self.after_epoch = after_epoch
    self.max_to_keep = max_to_keep

    self.set_model(model)

  def on_epoch_end(self, epoch, names, values):
    if epoch + 1 > self.after_epoch:
      result = {}
      for l in zip(names, values):
        result[l[0]] = l[1]
      super().on_epoch_end(epoch, logs=result)
      self._keep_fixed_number_models()

  def _keep_fixed_number_models(self):
    models = sorted(os.listdir(self.saved_path))
    if len(models) <= self.max_to_keep:
      return None
    else:
      for del_model in models[:-self.max_to_keep]:
        os.remove(os.path.join(self.saved_path, del_model))

  def _find_best_model_name(self):
    models = sorted(os.listdir(self.saved_path))
    if not len(models):
      return None
    best_model = models[-1]
    return os.path.join(self.saved_path, best_model)

  def get_init_epoch(self):
    best_model = self._find_best_model_name()
    if not best_model:
      return -1
    pattern = re.compile(r'.?epoch-(\d+)-.*')
    epoch = int(pattern.findall(best_model)[0])
    return epoch

  def restore_model(self, by_name=False):
    best_model = self._find_best_model_name()
    if not best_model:
      return None
    self.model.load_weights(best_model, by_name=by_name)
    logging.info('Loading weights: %s', best_model)


class ReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
  """keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                   verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

  """

  def __init__(self, model, monitor='val_loss', factor=0.1, patience=10,
               verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
               **kwargs):
    super().__init__(monitor=monitor, factor=factor, patience=patience,
                     verbose=verbose, mode=mode, min_delta=min_delta, cooldown=cooldown, min_lr=min_lr,
                     **kwargs)
    self.set_model(model=model)
    pass

  def on_epoch_end(self, epoch, names, values):
    assert len(names) == len(values)
    logs = {}
    for log in zip(names, values):
      logs[log[0]] = log[1]
    super().on_epoch_end(epoch=epoch, logs=logs)


class LearningRateScheduler(keras.callbacks.LearningRateScheduler):
  """keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                   verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

  """

  def __init__(self, model, schedule, verbose=0):
    super().__init__(schedule, verbose=0)
    self.set_model(model=model)
    pass

  # def on_epoch_end(self, epoch, names, values):
  #     assert len(names) == len(values)
  #     logs = {}
  #     for log in zip(names, values):
  #         logs[log[0]] = log[1]
  #     super().on_epoch_end(epoch=epoch, logs=logs)
