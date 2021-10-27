import time

import numpy as np
import logging, os
import datetime
import sys
from termcolor import colored

FORMAT = "[%(levelname)s]: %(message)s [%(name)s][%(filename)s:%(funcName)s():%(lineno)s][%(asctime)s.%(msecs)03d]"
DATEFMT = '%Y/%m/%d %H:%M:%S'


class _ColorfulFormatter(logging.Formatter):
  def __init__(self, *args, **kwargs):
    self._root_name = kwargs.pop("root_name") + "."
    self._abbrev_name = kwargs.pop("abbrev_name", "")
    if len(self._abbrev_name):
      self._abbrev_name = self._abbrev_name + "."
    super(_ColorfulFormatter, self).__init__(*args, **kwargs)

  def formatMessage(self, record):
    record.name = record.name.replace(self._root_name, self._abbrev_name)
    log = super(_ColorfulFormatter, self).formatMessage(record)
    if record.levelno == logging.WARNING:
      prefix = colored("WARNING", "red", attrs=["blink"])
    elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
      prefix = colored("ERROR", "red", attrs=["blink", "underline"])
    else:
      return log
    return prefix + " " + log


def logging_init(filename=None, level=logging.INFO, correct_time=False):
  def beijing(sec, what):
    '''sec and what is unused.'''
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

  if correct_time:
    logging.Formatter.converter = beijing

  logging.basicConfig(level=level,
                      format=FORMAT,
                      datefmt=DATEFMT,
                      filename=None, filemode='w')
  logger = logging.getLogger()

  # consoleHandler = logging.StreamHandler()
  # logger.addHandler(consoleHandler)

  if filename:
    logger_handler = logging.FileHandler(filename=filename, mode='w')
    logger_handler.setLevel(level=level)
    logger_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    logger.addHandler(logger_handler)

  def info_msg(*argv):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    logger.info(*argv)

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  logger.info_msg = info_msg
  return logger

  # logging.error('There are something wrong', exc_info=True)


def get_root_logger(filename, stream=True, level=logging.INFO):
  logger = logging.getLogger()
  logger.setLevel(level)
  set_hander(logger=logger, filename=filename, stream=stream, level=level)
  return logger


def get_logger(filename, logger_names=['template_lib', 'tl'], stream=True, level=logging.DEBUG, mode='a'):
  """

  :param filename:
  :param propagate: whether log to stdout
  :return:
  """

  logger_names += [filename, ]
  for name in logger_names:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    set_hander(logger=logger, filename=filename, stream=stream, level=level, mode=mode)
  return logger


def get_file_logger(filename, mode='w', logger_names=[], stream=False):
  logger = get_logger(filename=filename, logger_names=logger_names, stream=stream, mode=mode)
  return logger


def set_hander(logger, filename, stream=True, level=logging.INFO, mode='a'):
  formatter = logging.Formatter(
    "[%(asctime)s] %(name)s:%(lineno)s %(levelname)s: %(message)s \t[%(filename)s:%(funcName)s():%(lineno)s]",
    datefmt="%m/%d %H:%M:%S"
  )
  # formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

  file_hander = logging.FileHandler(filename=filename, mode=mode)
  file_hander.setLevel(level=level)
  file_hander.setFormatter(formatter)
  logger.addHandler(file_hander)

  def info_msg(*argv):
    # remove formats
    org_formatters = []
    for handler in logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))

    logger.info(*argv)

    # restore formats
    for handler, formatter in zip(logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  logger.info_msg = info_msg

  if stream:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    formatter = _ColorfulFormatter(
      # colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
      colored("[%(asctime)s] %(name)s %(levelname)s:", "blue") + \
      "%(message)s \t" + \
      colored("[%(filename)s:%(funcName)s():%(lineno)s]", "blue"),
      datefmt="%m/%d %H:%M:%S",
      root_name='template_lib',
      abbrev_name='tl',
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

  return logger


class StreamToLogger(object):
  """
  Fake file-like stream object that redirects writes to a logger instance.
  """

  def __init__(self, logger):
    self.logger = logger
    self.linebuf = ''

  def write(self, buf):
    buf = buf.rstrip('\n')
    if not buf:
      return
    buf = '<> ' + buf
    # for line in buf.rstrip().splitlines():
    #   self.logger.info_msg(line.rstrip())
    org_formatters = []
    for handler in self.logger.handlers:
      org_formatters.append(handler.formatter)
      handler.setFormatter(logging.Formatter("%(message)s"))
    self.logger.info(buf)
    # restore formats
    for handler, formatter in zip(self.logger.handlers, org_formatters):
      handler.setFormatter(formatter)

  def flush(self):
    pass

  def getvalue(self):
    pass

  def close(self):
    pass


def redirect_print_to_logger(logger, ):
  sl = StreamToLogger(logger)
  sys.stdout = sl
  sys.stderr = sl
  pass


def info_msg(logger, *argv):
  # remove formats
  org_formatters = []
  for handler in logger.handlers:
    org_formatters.append(handler.formatter)
    handler.setFormatter(logging.Formatter("%(message)s"))

  logger.info(*argv)

  # restore formats
  for handler, formatter in zip(logger.handlers, org_formatters):
    handler.setFormatter(formatter)