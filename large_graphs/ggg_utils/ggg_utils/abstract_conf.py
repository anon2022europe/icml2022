from typing import Dict
from copy import deepcopy

import attr

from ggg_utils.utils.utils import kwarg_create, filter_kwargs


@attr.s
class AbstractConf:
    def to_dict(self):
        return attr.asdict(deepcopy(self), recurse=True)

    @staticmethod
    def children() -> Dict:
        return dict()

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, dict):
            d = deepcopy(d)
            children = cls.children()
            for k in d.keys():
                if k in children:
                    d[k] = children[k].from_dict(d[k]) if d[k] is not None else None
            d = filter_kwargs(cls, d)
            d = cls(**d)
        return d

    def make(self):
        cls = type(self).OPTIONS[self.name]
        return kwarg_create(cls, self.to_dict())
