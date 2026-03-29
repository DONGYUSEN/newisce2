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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import logging
import os
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu

logger = logging.getLogger('isce.insar.runFilter')

def _validated_alpha(alpha, context):
    if alpha is None:
        raise ValueError('Filter strength is not set for {0}.'.format(context))

    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ValueError('Invalid filter strength "{0}" for {1}.'.format(alpha, context))

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(
            'Filter strength must be in [0, 1], got {0} for {1}.'.format(alpha, context)
        )

    if alpha < 0.5 or alpha > 0.8:
        logger.warning(
            'Filter strength %.3f for %s is outside recommended [0.5, 0.8] for topographic '
            'stripe suppression.',
            alpha,
            context,
        )

    return alpha

def _validate_interferogram(filename, context):
    xmlname = filename + '.xml'
    if (not os.path.isfile(filename)) or os.path.getsize(filename) == 0:
        raise RuntimeError(
            'Missing or empty interferogram "{0}" for {1}.'.format(filename, context)
        )

    if not os.path.isfile(xmlname):
        raise RuntimeError(
            'Missing interferogram metadata "{0}" for {1}.'.format(xmlname, context)
        )

    int_meta = isceobj.createIntImage()
    int_meta.load(xmlname)
    width = int_meta.getWidth()
    if (width is None) or (width <= 0):
        raise RuntimeError(
            'Invalid interferogram width ({0}) in "{1}" for {2}.'.format(
                width, xmlname, context
            )
        )
    return width

def _validate_output(filename, context):
    if (not os.path.isfile(filename)) or os.path.getsize(filename) == 0:
        raise RuntimeError(
            'Filtering produced missing/empty output "{0}" for {1}.'.format(filename, context)
        )

def runFilter(self, filterStrength):
    logger.info("Applying power-spectral filter")

    # Initialize the flattened interferogram
    topoflatIntFilename = self.insar.topophaseFlatFilename
    widthInt = _validate_interferogram(topoflatIntFilename, 'InsarProc.runFilter')
    intImage = isceobj.createIntImage()
    intImage.setFilename(topoflatIntFilename)
    intImage.setWidth(widthInt)
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtIntFilename = 'filt_' + topoflatIntFilename
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
    if filterStrength is not None:
        self.insar.filterStrength = filterStrength

    alpha = _validated_alpha(self.insar.filterStrength, 'InsarProc.runFilter')
    objFilter.goldsteinWerner(alpha=alpha)

    intImage.finalizeImage()
    filtImage.finalizeImage()
    _validate_output(filtIntFilename, 'InsarProc.runFilter')
    del filtImage
    
    #Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('read')
    filtImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType='FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(widthInt)
    phsigImage.setFilename(self.insar.phsigFilename)
    phsigImage.setAccessMode('write')
    phsigImage.setImageType('cor')#the type in this case is not for mdx.py displaying but for geocoding method
    phsigImage.createImage()

    
    ampImage = isceobj.createAmpImage()
    IU.copyAttributes(self.insar.resampAmpImage, ampImage)
    ampImage.setAccessMode('read')
    ampImage.createImage()


    icuObj = Icu(name='insarapp_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False

    icuObj.icu(intImage = filtImage, ampImage=ampImage, phsigImage=phsigImage)

    filtImage.finalizeImage()
    phsigImage.finalizeImage()
    ampImage.finalizeImage()
    phsigImage.renderHdr()



    # Set the filtered image to be the one geocoded
    self.insar.topophaseFlatFilename = filtIntFilename
