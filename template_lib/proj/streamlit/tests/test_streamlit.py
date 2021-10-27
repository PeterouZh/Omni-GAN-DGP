import os
import sys
import unittest

from template_lib import utils


class Testing_Streamlit(unittest.TestCase):

  def test_show_video(self, debug=True):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=7
        export TIME_STR=1
        export PYTHONPATH=./
        python -c "from template_lib.proj.streamlit.tests.test_streamlit import Testing_Streamlit;\
          Testing_Streamlit().test_show_video(debug=False)" \
          --tl_opts port 8530 start_web True show_video.num_video 7

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '0'
    from template_lib.v2.config_cfgnode.argparser import \
      (get_command_and_outdir, setup_outdir_and_yaml, get_append_cmd_str, start_cmd_run)

    tl_opts = ' '.join(sys.argv[sys.argv.index('--tl_opts') + 1:]) if '--tl_opts' in sys.argv else ''
    print(f'tl_opts:\n {tl_opts}')

    command, outdir = get_command_and_outdir(self, func_name=sys._getframe().f_code.co_name, file=__file__)
    argv_str = f"""
                --tl_config_file template_lib/proj/streamlit/scripts/configs/Streamlit.yaml
                --tl_command {command}
                --tl_outdir {outdir}
                --tl_opts {tl_opts}
                """
    args, cfg = setup_outdir_and_yaml(argv_str, return_cfg=True)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    script = "template_lib/proj/streamlit/scripts/run_web.py"
    if debug:
      cmd_str = f"""
          python 
            {script}
            {get_append_cmd_str(args)}
            --tl_debug
            --tl_opts
              """
    else:
      if cfg.start_web:
        cmd_str_prefix = f"""
                {os.path.dirname(sys.executable)}/streamlit run --server.port {cfg.port} 
                {script}
                --
              """
      else:
        cmd_str_prefix = f"python {script}"
      cmd_str = f"""
          {cmd_str_prefix}
            {get_append_cmd_str(args)}
            --tl_opts {tl_opts}
        """
    start_cmd_run(cmd_str)
    pass

