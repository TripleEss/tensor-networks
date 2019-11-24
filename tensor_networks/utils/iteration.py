from tensor_networks.utils.annotations import *

T = TypeVar('T')
def last(iterable: Iterable[T]) -> T:
    any_items = False
    for item in iterable:
        any_items = True
    if not any_items:
        raise StopIteration
    return item
