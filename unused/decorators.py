from tensor_networks.annotations import *


_RT = TypeVar('_RT')


def lazy_property(getter: Callable[..., _RT]) -> Callable[..., _RT]:
    attr_name = '_lazy_' + getter.__name__

    @property
    def _lazy_property(self) -> _RT:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, getter(self))
        return getattr(self, attr_name)

    return _lazy_property
