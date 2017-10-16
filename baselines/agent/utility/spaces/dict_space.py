from gym import Space
from collections import OrderedDict

class Dict(Space):
    """
    A dictionary of simpler spaces

    Example usage:
    self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
    """
    def __init__(self, spaces):
        if isinstance(spaces, dict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces

    def sample(self):
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __repr__(self):
        return "Dict(" + ", ". join([k + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret
