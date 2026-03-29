#
# Author: Piyush Agram
# Copyright 2016
#

import logging
import isceobj

from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
import os

logger = logging.getLogger('isce.topsinsar.runFilter')

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

def runFilter(self):

    if not self.doInSAR:
        return

    logger.info("Applying power-spectral filter")

    mergedir = self._insar.mergedDirname
    filterStrength = self.filterStrength
    context = 'TopsProc.runFilter'

    # Initialize the flattened interferogram
    inFilename = os.path.join(mergedir, self._insar.mergedIfgname)
    widthInt = _validate_interferogram(inFilename, context)
    intImage = isceobj.createIntImage()
    intImage.load(inFilename + '.xml')
    intImage.setAccessMode('read')
    intImage.createImage()

    # Create the filtered interferogram
    filtIntFilename = os.path.join(mergedir, self._insar.filtFilename)
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode('write')
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name='interferogram',object=intImage)
    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)

    alpha = _validated_alpha(filterStrength, context)
    objFilter.goldsteinWerner(alpha=alpha)

    intImage.finalizeImage()
    filtImage.finalizeImage()
    _validate_output(filtIntFilename, context)
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
    phsigImage.setFilename(os.path.join(mergedir, self._insar.coherenceFilename))
    phsigImage.setAccessMode('write')
    phsigImage.setImageType('cor')#the type in this case is not for mdx.py displaying but for geocoding method
    phsigImage.createImage()


    icuObj = Icu(name='topsapp_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False

    icuObj.icu(intImage = filtImage, phsigImage=phsigImage)

    filtImage.finalizeImage()
    phsigImage.finalizeImage()
    phsigImage.renderHdr()
