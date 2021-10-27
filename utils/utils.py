import errno
import os
import codecs
from typing import Any, Dict, Generic
from types import SimpleNamespace

import yaml


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Config:

    def _update_dic(self, dic, base_dic):
        """
        Update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def fromfile(self, path: str):
        return self._parse_from_yaml(path)

    def _parse_from_yaml(self, path: str):
        """Parse a yaml file and build config"""
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic


class DictWrapper:
    def __init__(self, dic):
        self._cfg_dict = dic
        self._cfg_obj = SimpleNamespace(**dic)

    def get_obj(self):
        return self._cfg_obj
