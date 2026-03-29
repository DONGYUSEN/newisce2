#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from .Dem1Manager import Dem1Manager
from iscesys.Component.Component import Component
import numpy as np
from isceobj.Image import createDemImage

STEP_SRTMGL1_URL = 'https://step.esa.int/auxdata/dem/SRTMGL1'

EXTRA = Component.Parameter('_extra',
    public_name = 'extra',default = '.SRTMGL1',
    type = str,
    mandatory = False,
    doc = 'String to append to default name such as .SRTMGL1 for dem. Since the default is set to read usgs' \
          +' dems if extra is empty one needs to enter a empty string "" in the xml file' \
          +' otherwise if no value is provided is then interpreted as None by the xml reader.')

URL = Component.Parameter('_url',
    public_name = 'URL',default = STEP_SRTMGL1_URL,
    type = str,
    mandatory = False,
    doc = "Url for the high resolution DEM.")

TILE_SIZE = Component.Parameter('_tileSize',
    public_name = 'tileSize',
    default = [3601,3601],
    container=list,
    type=int,
    mandatory = True,
    doc = 'Two element list with the number of row and columns of the tile.')

##Base class to handle product such as dem or water mask
class Dem3Manager(Dem1Manager):
    family = 'dem1manager'
    parameter_list = (
                       EXTRA,
                       URL,
                       TILE_SIZE
                       ) + Dem1Manager.parameter_list


    def __init__(self,family = '', name = ''):
        self.parameter_list = self.parameter_list + super(Dem1Manager,self).parameter_list
        self.updateParameters()
        super(Dem3Manager, self).__init__(family if family else  self.__class__.family, name=name)
        self._tileWidth = 3600
    def updateParameters(self):
        self.extendParameterList(Dem1Manager,Dem3Manager)
        super(Dem3Manager,self).updateParameters()
