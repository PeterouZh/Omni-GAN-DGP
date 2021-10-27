import copy
from typing import Dict, Optional
import logging
from fvcore.common.registry import Registry as Registry_base


class Registry(Registry_base):

  def _do_register(self, name: str, obj: object) -> None:
    assert (
          name not in self._obj_map
    ), "An object named '{}' was already registered in '{}' registry!".format(
      name, self._name
    )
    # logging.getLogger('tl').warning(f"\n  {name} was already registered in {self._name} registry!")
    self._obj_map[name] = obj
    pass

  def register(self, obj: object = None, name=None, name_prefix=None) -> Optional[object]:
    """
    Register the given object under the the name `obj.__name__`.
    Can be used as either a decorator or not. See docstring of this class for usage.
    """
    if obj is None:
      # used as a decorator
      def deco(func_or_class: object, name=name, name_prefix=name_prefix) -> object:
        # name = func_or_class.__name__  # pyre-ignore
        if name is None:
          if name_prefix is None:
            name = f"{func_or_class.__module__}.{func_or_class.__name__}"
          else:
            name = f"{name_prefix}.{func_or_class.__name__}"
        else:
          name = name
        self._do_register(name, func_or_class)
        return func_or_class

      return deco


def build_from_cfg(cfg, registry, kwargs_priority, default_args=None):

  if not isinstance(cfg, dict):
    raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
  if 'name' not in cfg:
    if default_args is None or 'type' not in default_args:
      raise KeyError(
        '`cfg` or `default_args` must contain the key "type", '
        f'but got {cfg}\n{default_args}')
  if not isinstance(registry, Registry):
    raise TypeError('registry must be an Registry object, '
                    f'but got {type(registry)}')
  if not (isinstance(default_args, dict) or default_args is None):
    raise TypeError('default_args must be a dict or None, '
                    f'but got {type(default_args)}')

  # cfg = cfg.clone()
  cfg = copy.deepcopy(cfg)
  if kwargs_priority:
    args = {**cfg, **default_args}
  else:
    args = {**default_args, **cfg}

  obj_type = args.pop('name')
  obj_cls = registry.get(obj_type)
  return obj_cls(**args)