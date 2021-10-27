import os, sys
import pprint


class TensorBoardTool(object):
  """ Run tensorboard in python
      Usage:
          from peng_lib.torch_utils import TensorBoardTool
          tbtool = TensorBoardTool(dir_path=args.tbdir)
          writer = tbtool.writer
  """

  def __init__(self, tbdir):
    self.tbdir = tbdir
    # create tensorboardx writer
    writer = self.SummmaryWriter()
    self.writer = writer


  def run(self):
    """ Launch tensorboard and create summary_writer

    :return:
    """
    import logging
    from tensorboard import default
    from tensorboard import program

    # Remove http messages
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # Start tensorboard server
    tb = program.TensorBoard(default.get_plugins())
    # tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
    port = os.getenv('PORT', '6006')

    tb.configure(argv=[None, '--logdir', self.tbdir, '--port', port])
    url = tb.launch()
    sys.stdout.write('TensorBoard at %s \n' % url)

    return

  def SummmaryWriter(self):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(logdir=self.tbdir)
    return writer

  def add_text_str_args(self, args, name):
    args_str = pprint.pformat(args)
    args_str = args_str.strip().replace('\n', '  \n>')
    self.writer.add_text(name, args_str, 0)

  def add_text_md_args(self, args, name):
    args_md = self.args_as_markdown_no_sorted_(args)
    self.writer.add_text(name, args_md, 0)

  def add_text_args_and_od(self, args, od):
    od_md = self.args_as_markdown_no_sorted_(od)
    self.writer.add_text('od', od_md, 0)

    default_args = set(vars(args)) - set(od)
    default_args = {key: vars(args)[key] for key in default_args}
    default_args_md = self.args_as_markdown_sorted_(default_args)
    self.writer.add_text('args', default_args_md, 0)

  @staticmethod
  def args_as_markdown_no_sorted_(args):
    """ Return configs as markdown format

    :param args: dict
    :return:
    """
    text = "|name|value|  \n|:-:|:-:|  \n"
    for attr, value in args.items():
      text += "|{}|{}|  \n".format(attr, value)
    return text

  @staticmethod
  def args_as_markdown_sorted_(args):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(args.items()):
      text += "|{}|{}|  \n".format(attr, value)
    return text
