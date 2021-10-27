import sys, os, logging
from collections import OrderedDict
import tensorflow as tf
from tensorboard import default
from tensorboard import program

import time
import datetime
import numpy as np
from subprocess import check_output

FLAGS = tf.app.flags.FLAGS


def get_define_flags():
  """ FLAGS.flag_values_dict()
  Usage:
      import peng_lib.tf_utils
      DEFINE_flag = peng_lib.tf_utils.get_define_flags()
      od = DEFINE_flag.od
      tf.app.run(main=train.main, argv=[od])
  Get file and func name:
      DEFINE_flag('python_file', __file__, '')
      DEFINE_flag('func_call', sys._getframe().f_code.co_name, '')
  :return:
  """
  od = OrderedDict()

  def DEFINE_flag(key, value, help=None):
    od[key] = value
    # remove defined flag
    if key in FLAGS:
      FLAGS.remove_flag_values({key: None})
    if isinstance(value, str):
      tf.app.flags.DEFINE_string(key, value, help)
    elif isinstance(value, bool):
      tf.app.flags.DEFINE_boolean(key, value, help)
    elif isinstance(value, int):
      tf.app.flags.DEFINE_integer(key, value, help)
    elif isinstance(value, float):
      tf.app.flags.DEFINE_float(key, value, help)
    elif isinstance(value, list) and isinstance(value[0], int):
      tf.app.flags.DEFINE_multi_integer(key, value, help)
    else:
      assert 0

  DEFINE_flag.od = od
  return DEFINE_flag


class TensorBoardTool:
  """ Run tensorboard in python

  """

  def __init__(self, dir_path):
    self.dir_path = dir_path

  def run_v1_10(self):
    # Remove http messages
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    # Start tensorboard server
    tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
    tb.configure(argv=['--logdir', self.dir_path])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)

  def run(self):
    # Remove http messages
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # Start tensorboard server
    tb = program.TensorBoard(default.get_plugins())
    # tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
    port = os.getenv('PORT')
    if not port:
      logging.warning("Don't set tensorboard port, use 6006!")
      port = '6006'
    tb.configure(argv=[None, '--logdir', self.dir_path, '--port', port])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)


def create_dirs(dirs):
  """
  dirs - a list of directories to create if these directories are not found
  :param dirs:
  :return exit_code: 0:success -1:failed
  """
  try:
    for dir_ in dirs:
      if not os.path.exists(dir_):
        os.makedirs(dir_)
    return 0
  except Exception as err:
    print("Creating directories error: {0}".format(err))
    exit(-1)


class Logger:
  def __init__(self, sess, summary_dir, log_graph=True):
    self.sess = sess
    self.summary_dir = summary_dir

    self.summary_writer = {}
    self.summary_placeholders = {}
    self.summary_ops = {}

    create_dirs([self.summary_dir, ])
    graph = None
    if log_graph:
      graph = sess.graph
    self.summary_writer['main'] = tf.summary.FileWriter(os.path.join(self.summary_dir, 'main'), graph)
    print(" [*] Tensorboard logging in %s" % self.summary_dir)

    # save log dir and ip
    ips = check_output(['hostname', '--all-ip-addresses'])
    ips = ips.replace(b' ', b'  \n>')
    logdir_ip = ('Log_dir: ' + self.summary_dir + ' \n>').encode('utf-8') + ips
    logdir_ip = str(logdir_ip, encoding='utf-8')
    self.summarize(summaries_dict={'Log_dir_and_ip': logdir_ip}, step=0, type='text', curent_time=True)
    pass

  def create_no_graph_summary_writer(self, name, summary_dir):
    self.summary_writer[name] = tf.summary.FileWriter(summary_dir)

  def summarize_config_dict_to_string(self, config_flags):
    import json
    config_str = json.dumps(config_flags, indent=2)
    # for pretty indent
    config_str = config_str.replace('\n', '  \n>')
    self.summarize(summaries_dict={'Config': config_str}, step=0, type='text', summary_name='main')

  # it can summarize scalars and images.
  def summarize(self, summaries_dict, step, type, max_image_outputs=4, summary_name=None, curent_time=False):
    """

    :param summaries_dict: the dict of the summaries values (tag,value)
    :param step: the step of the summary
    :param type: [scalar, image, hist, text]
    :param max_image_outputs:
    :param summary_name: which summay_writer
    :param curent_time: add time to text summary

    :return:
    """
    # summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
    if not summary_name:
      scope = 'main'
      summary_writer = self.summary_writer['main']
    else:
      scope = summary_name
      summary_writer = self.summary_writer[summary_name]

    with tf.variable_scope(scope):
      if summaries_dict is not None:
        summary_list = []
        for tag, value in summaries_dict.items():
          # create op
          if tag not in self.summary_ops:
            value = np.array(value)
            if type == 'scalar':
              self.summary_placeholders[tag] = tf.placeholder('float32', value.shape,
                                                              name='{0}_pl'.format(tag))
              self.summary_ops[tag] = tf.summary.scalar('{0}_scalar'.format(tag), self.summary_placeholders[tag])
            elif type == 'image':
              # summary images
              self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]),
                                                              name='{0}_pl'.format(tag))
              self.summary_ops[tag] = tf.summary.image('{0}_image'.format(tag), self.summary_placeholders[tag],
                                                       max_outputs=max_image_outputs)
            elif type == 'hist':
              self.summary_placeholders[tag] = tf.placeholder('float32', value.shape,
                                                              name='{0}_pl'.format(tag))
              self.summary_ops[tag] = tf.summary.histogram('{0}_hist'.format(tag),
                                                           self.summary_placeholders[tag])
            elif type == 'text':
              self.summary_placeholders[tag] = tf.placeholder(tf.string, value.shape,
                                                              name='{0}_pl'.format(tag))
              self.summary_ops[tag] = tf.summary.text('{0}_text'.format(tag),
                                                      self.summary_placeholders[tag])
            else:
              assert 0

          if type == 'text' and curent_time:
            time_str = str((datetime.datetime.utcnow() + datetime.timedelta(hours=8))
                           .strftime('%y-%m-%d %H:%M:%S'))
            value = str(value) + '  [%s]' % time_str

          summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

        for summary in summary_list:
          summary_writer.add_summary(summary, step)
        summary_writer.flush()
