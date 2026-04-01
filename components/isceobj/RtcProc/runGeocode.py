#!/usr/bin/env python3
# coding: utf-8
# Author: Simon Kraatz
# Copyright 2016

import logging
import isceobj
import os
import numpy as np
import math
from isceobj.Util.decorators import use_api
from osgeo import gdal, ogr, osr

logger = logging.getLogger('isce.grdsar.looks')

def _estimate_square_multilook_spacing(self, gamma_vrt_path):
    '''
    Estimate multilook spacing in meters and force square integer spacing:
    ceil(max(xres_ml, yres_ml)).
    '''
    ref_pol = self._grd.polarizations[0]
    reference = self._grd.loadProduct(
        os.path.join(self._grd.outputFolder, 'beta_{0}.xml'.format(ref_pol))
    )

    ds = gdal.OpenShared(gamma_vrt_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError('Cannot open gamma VRT for spacing estimation: {0}'.format(gamma_vrt_path))

    out_width = int(ds.RasterXSize)
    out_length = int(ds.RasterYSize)
    ds = None
    if out_width <= 0 or out_length <= 0:
        raise RuntimeError('Invalid output size from gamma VRT: {0} x {1}'.format(out_width, out_length))

    in_width = float(reference.numberOfSamples)
    in_length = float(reference.numberOfLines)
    if in_width <= 0.0 or in_length <= 0.0:
        raise RuntimeError('Invalid reference product dimensions.')

    rg_factor = in_width / float(out_width)
    az_factor = in_length / float(out_length)
    xres_ml = float(reference.groundRangePixelSize) * rg_factor
    yres_ml = float(reference.azimuthPixelSize) * az_factor

    spacing = max(1, int(math.ceil(max(xres_ml, yres_ml))))
    return float(spacing), float(xres_ml), float(yres_ml), float(rg_factor), float(az_factor)

def runGeocode(self):
    '''
    Geocode a swath file using corresponding lat, lon files
    '''
    sourcexmltmpl = '''    <SimpleSource>
      <SourceFilename>{0}</SourceFilename>
      <SourceBand>{1}</SourceBand>
    </SimpleSource>'''
    
    gcl = [f for f in os.listdir(self._grd.outputFolder) if f.startswith('gamma') and f.endswith('.vrt')] 
    a, b = os.path.split(self._grd.outputFolder)
    latfile = os.path.join(a,self._grd.geometryFolder,'lat.rdr.vrt')
    lonfile = os.path.join(a,self._grd.geometryFolder,'lon.rdr.vrt')
    
    outsrs = 'EPSG:'+str(self.epsg)
    gspacing = self.gspacing
    method = self.intmethod
    insrs = 4326 
    fmt = 'GTiff'
    fl = len(gcl)

    if (gspacing is None) or (float(gspacing) <= 0.0):
        if fl > 0:
            gamma_vrt = os.path.join(a, self._grd.outputFolder, gcl[0])
            try:
                gspacing, xres_ml, yres_ml, rg_factor, az_factor = _estimate_square_multilook_spacing(self, gamma_vrt)
                print(
                    'INFO: auto gspacing from multilook resolution: '
                    'xres_ml={0:.3f} m, yres_ml={1:.3f} m, '
                    'rangeFactor={2:.3f}, azFactor={3:.3f}, '
                    'square integer spacing={4:.0f} m'.format(
                        xres_ml, yres_ml, rg_factor, az_factor, gspacing
                    )
                )
            except Exception as err:
                fallback = max(1, int(math.ceil(float(self.posting)))) if self.posting else 10
                gspacing = float(fallback)
                logger.warning(
                    'Auto gspacing estimation failed (%s); fallback to posting-based spacing=%s m',
                    err,
                    gspacing
                )
        else:
            fallback = max(1, int(math.ceil(float(self.posting)))) if self.posting else 10
            gspacing = float(fallback)
            logger.warning('No gamma VRT files found; fallback gspacing=%s m', gspacing)
    else:
        gspacing = float(gspacing)
    
    for num, val in enumerate(gcl):
        print('****Geocoding file %s out of %s: %s****' %(num+1, fl, val))
        infile = os.path.join(a, self._grd.outputFolder, val)
        outfile = os.path.join(a, self._grd.outputFolder, val[:-3]+'tif')
        
        driver = gdal.GetDriverByName('VRT')
        tempvrtname = os.path.join(a, self._grd.outputFolder, 'geocode.vrt')

        inds = gdal.OpenShared(infile, gdal.GA_ReadOnly)
        tempds = driver.Create(tempvrtname, inds.RasterXSize, inds.RasterYSize, 0)

        for ii in range(inds.RasterCount):
            band = inds.GetRasterBand(1)
            tempds.AddBand(band.DataType)
            tempds.GetRasterBand(ii+1).SetMetadata({'source_0': sourcexmltmpl.format(infile, ii+1)}, 'vrt_sources')
      
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(insrs)
        srswkt = sref.ExportToWkt()
        
        tempds.SetMetadata({'SRS' : srswkt,
                            'X_DATASET': lonfile,
                            'X_BAND' : '1',
                            'Y_DATASET': latfile,
                            'Y_BAND' : '1',
                            'PIXEL_OFFSET' : '0',
                            'LINE_OFFSET' : '0',
                            'PIXEL_STEP' : '1',
                            'LINE_STEP' : '1'}, 'GEOLOCATION')
        
        band = None
        tempds = None 
        inds = None
        bounds = None
        
        spacing = [gspacing, gspacing]
        
        warpOptions = gdal.WarpOptions(format=fmt,
                                       xRes=spacing[0], yRes=spacing[1],
                                       dstSRS=outsrs,
                                       outputBounds = bounds,
                                       resampleAlg=method, geoloc=True)
        gdal.Warp(outfile, tempvrtname, options=warpOptions)
    
    return
