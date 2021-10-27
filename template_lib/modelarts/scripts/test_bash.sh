

gpu=$1
#echo `pwd`
export PYTHONPATH=./
#which python
export MKL_THREADING_LAYER=
python -c "from template_lib.examples import test_bash; \
  test_bash.TestingUnit().test_resnet(gpu='$gpu')"





