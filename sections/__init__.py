"""Pişirme Laboratuvarı analysis sections."""

from .potato import run_potato
from .pizza import run_pizza
from .borek import run_borek
from .smallcake import run_smallcake
from .pyrocam import run_pyrocam
from .bread import run_bread_surface
from .data_merger import run_data_merger
from .teflon_block import run_teflon_block
from .cookie import run_cookie
from .flour_disk import run_flour_disk
from .toast import run_toast

__all__ = [
    "run_potato",
    "run_pizza",
    "run_borek",
    "run_smallcake",
    "run_pyrocam",
    "run_bread_surface",
    "run_data_merger",
    "run_teflon_block",
    "run_cookie",
    "run_flour_disk",
    "run_toast",
]
