from template_lib.v2.GAN.evaluation.tf_FID_IS_score import TFFIDISScore

from template_lib.v2.config_cfgnode import update_parser_defaults_from_yaml, build_parser

if __name__ == '__main__':

  parser = update_parser_defaults_from_yaml(parser=None, use_cfg_as_args=True)
  args = parser.parse_args()

  TFFIDISScore.test_case_calculate_fid_stat_CIFAR()
  pass
