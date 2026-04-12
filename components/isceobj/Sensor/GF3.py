#!/usr/bin/env python3

"""GF3 sensor adapter.

Current implementation reuses the Lutan1 stripmap parser interface so that
GF3 can participate in the same high-resolution stripmap workflow. This keeps
an independent class boundary for future GF3-specific metadata handling.
"""

from .Lutan1 import Lutan1


class GF3(Lutan1):
    family = 'gf3sm'
    logging_name = 'isce.sensor.GF3'

    def __init__(self, name=''):
        super(GF3, self).__init__(name=name)
