#!/usr/bin/env python3

"""DJ1 sensor adapter.

Current implementation reuses the Tianyi stripmap parser interface and keeps
an independent class for future DJ1-specific specialization.
"""

from .Tianyi import Tianyi


class DJ1(Tianyi):
    family = 'dj1sm'
    logging_name = 'isce.sensor.DJ1'

    def __init__(self, family='', name=''):
        super(DJ1, self).__init__(family if family else self.__class__.family, name=name)
