import os
import sys
import argparse
import shutil


root_obs_dict = {
  'beijing': 's3://bucket-cv-competition-bj4/ZhouPeng',
  'huanan': 's3://bucket-1892/ZhouPeng',
  'huabei': 's3://bucket-cv-competition/ZhouPeng',

}

parser = argparse.ArgumentParser()
parser.add_argument('-ro', '--root-obs', '--root_obs', type=str, default=None, choices=list(root_obs_dict.keys()))
parser.add_argument('--train_url', default=None)
parser.add_argument('--port', type=int, default=6001)
parser.add_argument('--env', type=str, default='')
parser.add_argument('--exp', type=str, default='')


def setup_package():
  packages = ['pyyaml==5.2', 'easydict', 'tensorboardX==1.9', 'gitpython']
  command_template = 'pip install %s'
  for pack in packages:
    command = command_template % pack
    print('=Installing %s'%pack)
    os.system(command)
  # install git
  print('=Installing git')
  os.system('python /cache/code/template_lib/examples/copy_tool.py \
            -s {root_obs}/pypi/template_lib -d /cache/pypi/template_lib -t copytree'.format(root_obs=os.environ['ROOT_OBS']))
  os.system('cd /cache/pypi/template_lib && \
            sudo dpkg -i liberror-perl_0.17-1.2_all.deb && \
            sudo dpkg -i git-man_2.7.4-0ubuntu1_all.deb && \
            sudo dpkg -i git_2.7.4-0ubuntu1_amd64.deb')

def setup_env(root_obs, train_url=None, **kwargs):
  os.environ['ROOT_OBS'] = root_obs
  os.environ['RESULTS_OBS'] = os.path.join(root_obs, 'results/template_lib')
  if 'DLS_TRAIN_URL' not in os.environ:
    if train_url is not None:
      os.environ['DLS_TRAIN_URL'] = train_url
    else:
      os.environ['DLS_TRAIN_URL'] = '/tmp/logs/1'

def setup_dir():
  home_dir = os.path.expanduser('~')
  os.system('rm %s'%(os.path.join(home_dir, '.keras')))
  os.system('mkdir /cache/.keras')
  os.system('ln -s /cache/.keras %s'%home_dir)


if __name__ == '__main__':
  # args = parser.parse_args()
  args, unparsed = parser.parse_known_args()
  args.root_obs = root_obs_dict[args.root_obs]

  setup_env(**vars(args))
  setup_dir()

  try:
    import moxing
    code_obs = os.path.join(args.root_obs, 'code')
    code = '/cache/code'
    print('Copying code from [%s] to [%s]'%(code_obs, code))
    moxing.file.copy_parallel(code_obs, code)
    # print('Convert all bash files to unix end of line.')
    # os.system("find /cache/code -name '*.sh' | xargs perl -pi -e 's/\r\n/\n/g'")
    # print('End convert all bash files to unix end of line.')
  except ImportError:
    pass
  except Exception as e:
    if str(e) == 'server is not set correctly':
      pass
    else:
      raise e
  finally:
    os.chdir(os.path.join(code, 'template_lib/examples'))

  cwd = os.getcwd()
  print('cwd: %s'%cwd)

  if 'PORT' in os.environ and os.environ['PORT'] == str(args.port):
    assert 0, args.port
  
  setup_package()

  command = '''
        export CUDA_VISIBLE_DEVICES=0
        export PORT={port}
        export TIME_STR=1
        export PYTHONPATH=../..
        python -c "import test_bash; \
          test_bash.TestingUnit().test_bash($PORT)"
        '''.format(port=args.port)

  try:
    os.system("sed -i.bak 's/\r$//' /cache/code/template_lib/examples/test_bash.sh")
    os.system(command)
  except:
    pass
  pass
