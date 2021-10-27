

gpu=$1
export PYTHONPATH=../..
python -c "import test_bash; \
  test_bash.TestingUnit().test_resnet(gpu='$gpu')"