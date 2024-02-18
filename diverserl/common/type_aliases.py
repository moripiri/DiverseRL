from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

T = TypeVar("T")
_t_or_tuple_2_t = Union[T, Tuple[T, T]]
_t_or_tuple_any_t = Union[T, Tuple[T, ...]]
_int_or_tuple_2_int = _t_or_tuple_2_t[int]
_int_or_tuple_any_int = _t_or_tuple_any_t[int]
_str_or_tuple_any_str = _t_or_tuple_any_t[str]
_ist_or_tuple_any_ist = _t_or_tuple_any_t[Union[int, str, Tuple]]

_layers_size_any_int = _t_or_tuple_any_t[_int_or_tuple_2_int]
_layers_size_any_si = _t_or_tuple_any_t[Union[str, _int_or_tuple_2_int]]
