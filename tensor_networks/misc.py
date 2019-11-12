from utils.annotations import *

class ArrayTuple(Tuple[ndarray]):
    @property
    def shape(self):
        return tuple(arr.shape for arr in self)

    def __add__(self, other) -> 'ArrayTuple':
        return type(self)(super().__add__(other))

    def __radd__(self, other) -> 'ArrayTuple':
        return type(self)(super(type(self), type(self)).__add__(other, self))

    def __mul__(self, other) -> 'ArrayTuple':
        return type(self)(super().__mul__(other))

    __rmul__ = __mul__

    def count(self, x) -> int:
        c = 0
        for arr in self:
            if x is arr or (x == arr).all():
                c += 1
        return c

    def index(self, x, start: int = 0, end: int = -1) -> int:
        for i, arr in enumerate(self[start:end]):
            if x is arr or (x == arr).all():
                return i
        raise ValueError(f'{type(self).__name__}.index(x):'
                         f' x not in {type(self).__name__}')

    def __contains__(self, item) -> bool:
        try:
            self.index(item)
        except ValueError:
            return False
        return True

    def __repr__(self) -> str:
        return type(self).__name__ + super().__repr__()
