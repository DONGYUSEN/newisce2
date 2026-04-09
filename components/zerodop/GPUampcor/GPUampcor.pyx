#
# Author: Joshua Cohen
# Copyright 2016
#

from libc.stdint cimport uint64_t
from libcpp cimport bool

from mroipac.ampcor import isce3_engine


cdef extern from "Ampcor.h":
    cdef cppclass Ampcor:
        uint64_t imgAccessor1, imgAccessor2, offImgAccessor, offQualImgAccessor
        float snrThresh, covThresh, xScaleFactor, yScaleFactor
        int imgDatatypes[2]
        int imgWidths[2]
        int imgBands[2]
        int isMag[2]
        int firstRow, lastRow, rowSkip, firstCol, lastCol, colSkip, refChipWidth, refChipHeight
        int schMarginX, schMarginY, nLookAcross, nLookDown, osampFact, zoomWinSize
        int acrossGrossOff, downGrossOff, numRowTable
        bool corr_debug, corr_display, usr_enable_gpu
        
        Ampcor() except +
        void ampcor()
        int getLocationAcrossAt(int)
        int getLocationDownAt(int)
        float getLocationAcrossOffsetAt(int)
        float getLocationDownOffsetAt(int)
        float getSnrAt(int)
        float getCov1At(int)
        float getCov2At(int)
        float getCov3At(int)


cdef class PyAmpcor:
    cdef Ampcor c_ampcor
    cdef object _engine
    cdef object _referenceImageName
    cdef object _secondaryImageName
    cdef int _imageLength1
    cdef int _imageLength2
    cdef list _locationAcrossPy
    cdef list _locationDownPy
    cdef list _locationAcrossOffsetPy
    cdef list _locationDownOffsetPy
    cdef list _snrPy
    cdef list _cov1Py
    cdef list _cov2Py
    cdef list _cov3Py
    cdef int _numElemPy
    
    def __cinit__(self):
        self._engine = isce3_engine.ENGINE_LEGACY
        self._referenceImageName = None
        self._secondaryImageName = None
        self._imageLength1 = -1
        self._imageLength2 = -1
        self._locationAcrossPy = []
        self._locationDownPy = []
        self._locationAcrossOffsetPy = []
        self._locationDownOffsetPy = []
        self._snrPy = []
        self._cov1Py = []
        self._cov2Py = []
        self._cov3Py = []
        self._numElemPy = 0
        return
    
    @property
    def imageBand1(self):
        return self.c_ampcor.imgBands[0]
    @imageBand1.setter
    def imageBand1(self, int a):
        self.c_ampcor.imgBands[0] = a
    @property
    def imageBand2(self):
        return self.c_ampcor.imgBands[1]
    @imageBand2.setter
    def imageBand2(self, int a):
        self.c_ampcor.imgBands[1] = a
    @property
    def imageAccessor1(self):
        return self.c_ampcor.imgAccessor1
    @imageAccessor1.setter
    def imageAccessor1(self, uint64_t a):
        self.c_ampcor.imgAccessor1 = a
    @property
    def imageAccessor2(self):
        return self.c_ampcor.imgAccessor2
    @imageAccessor2.setter
    def imageAccessor2(self, uint64_t a):
        self.c_ampcor.imgAccessor2 = a
    @property
    def engine(self):
        return self._engine
    @engine.setter
    def engine(self, a):
        self._engine = isce3_engine.normalize_engine(a)
    @property
    def referenceImageName(self):
        return self._referenceImageName
    @referenceImageName.setter
    def referenceImageName(self, a):
        self._referenceImageName = a
    @property
    def secondaryImageName(self):
        return self._secondaryImageName
    @secondaryImageName.setter
    def secondaryImageName(self, a):
        self._secondaryImageName = a
    @property
    def offsetImageAccessor(self):
        return self.c_ampcor.offImgAccessor
    @offsetImageAccessor.setter
    def offsetImageAccessor(self, uint64_t a):
        self.c_ampcor.offImgAccessor = a
    @property
    def offsetQualImageAccessor(self):
        return self.c_ampcor.offQualImgAccessor
    @offsetQualImageAccessor.setter
    def offsetQualImageAccessor(self, uint64_t a):
        self.c_ampcor.offQualImgAccessor = a
    @property
    def thresholdSNR(self):
        return self.c_ampcor.snrThresh
    @thresholdSNR.setter
    def thresholdSNR(self, float a):
        self.c_ampcor.snrThresh = a
    @property
    def thresholdCov(self):
        return self.c_ampcor.covThresh
    @thresholdCov.setter
    def thresholdCov(self, float a):
        self.c_ampcor.covThresh = a
    @property
    def scaleFactorX(self):
        return self.c_ampcor.xScaleFactor
    @scaleFactorX.setter
    def scaleFactorX(self, float a):
        self.c_ampcor.xScaleFactor = a
    @property
    def scaleFactorY(self):
        return self.c_ampcor.yScaleFactor
    @scaleFactorY.setter
    def scaleFactorY(self, float a):
        self.c_ampcor.yScaleFactor = a
    @property
    def datatype1(self):
        dt = self.c_ampcor.imgDatatypes[0]
        mg = self.c_ampcor.isMag[0]
        if (dt + mg == 0):
            return 'real'
        elif (dt + mg == 1):
            return 'complex'
        else: # dt + mg == 2
            return 'mag'
    @datatype1.setter
    def datatype1(self, str a):
        if (a[0].lower() == 'r'):
            self.c_ampcor.isMag[0] = 0
            self.c_ampcor.imgDatatypes[0] = 0
        elif (a[0].lower() == 'c'):
            self.c_ampcor.isMag[0] = 0
            self.c_ampcor.imgDatatypes[0] = 1
        elif (a[0].lower() == 'm'):
            self.c_ampcor.isMag[0] = 1
            self.c_ampcor.imgDatatypes[0] = 1
        else:
            print("Error: Unrecognized datatype. Expected 'complex', 'real', or 'mag'.")
    @property
    def datatype2(self):
        dt = self.c_ampcor.imgDatatypes[1]
        mg = self.c_ampcor.isMag[1]
        if (dt + mg == 0):
            return 'real'
        elif (dt + mg == 1):
            return 'complex'
        else: # dt + mg == 2
            return 'mag'
    @datatype2.setter
    def datatype2(self, str a):
        if (a[0].lower() == 'r'):
            self.c_ampcor.isMag[1] = 0
            self.c_ampcor.imgDatatypes[1] = 0
        elif (a[0].lower() == 'c'):
            self.c_ampcor.isMag[1] = 0
            self.c_ampcor.imgDatatypes[1] = 1
        elif (a[0].lower() == 'm'):
            self.c_ampcor.isMag[1] = 1
            self.c_ampcor.imgDatatypes[1] = 1
        else:
            print("Error: Unrecognized datatype. Expected 'complex', 'real', or 'mag'.")
    @property
    def lineLength1(self):
        return self.c_ampcor.imgWidths[0]
    @lineLength1.setter
    def lineLength1(self, int a):
        self.c_ampcor.imgWidths[0] = a
    @property
    def lineLength2(self):
        return self.c_ampcor.imgWidths[1]
    @lineLength2.setter
    def lineLength2(self, int a):
        self.c_ampcor.imgWidths[1] = a
    @property
    def imageLength1(self):
        return self._imageLength1
    @imageLength1.setter
    def imageLength1(self, int a):
        self._imageLength1 = a
    @property
    def imageLength2(self):
        return self._imageLength2
    @imageLength2.setter
    def imageLength2(self, int a):
        self._imageLength2 = a
    @property
    def firstSampleDown(self):
        return self.c_ampcor.firstRow
    @firstSampleDown.setter
    def firstSampleDown(self, int a):
        self.c_ampcor.firstRow = a
    @property
    def lastSampleDown(self):
        return self.c_ampcor.lastRow
    @lastSampleDown.setter
    def lastSampleDown(self, int a):
        self.c_ampcor.lastRow = a
    @property
    def skipSampleDown(self):
        return self.c_ampcor.rowSkip
    @skipSampleDown.setter
    def skipSampleDown(self, int a):
        self.c_ampcor.rowSkip = a
    @property
    def firstSampleAcross(self):
        return self.c_ampcor.firstCol
    @firstSampleAcross.setter
    def firstSampleAcross(self, int a):
        self.c_ampcor.firstCol = a
    @property
    def lastSampleAcross(self):
        return self.c_ampcor.lastCol
    @lastSampleAcross.setter
    def lastSampleAcross(self, int a):
        self.c_ampcor.lastCol = a
    @property
    def skipSampleAcross(self):
        return self.c_ampcor.colSkip
    @skipSampleAcross.setter
    def skipSampleAcross(self, int a):
        self.c_ampcor.colSkip = a
    @property
    def windowSizeWidth(self):
        return self.c_ampcor.refChipWidth
    @windowSizeWidth.setter
    def windowSizeWidth(self, int a):
        self.c_ampcor.refChipWidth = a
    @property
    def windowSizeHeight(self):
        return self.c_ampcor.refChipHeight
    @windowSizeHeight.setter
    def windowSizeHeight(self, int a):
        self.c_ampcor.refChipHeight = a
    @property
    def searchWindowSizeWidth(self):
        return self.c_ampcor.schMarginX
    @searchWindowSizeWidth.setter
    def searchWindowSizeWidth(self, int a):
        self.c_ampcor.schMarginX = a
    @property
    def searchWindowSizeHeight(self):
        return self.c_ampcor.schMarginY
    @searchWindowSizeHeight.setter
    def searchWindowSizeHeight(self, int a):
        self.c_ampcor.schMarginY = a
    @property
    def acrossLooks(self):
        return self.c_ampcor.nLookAcross
    @acrossLooks.setter
    def acrossLooks(self, int a):
        self.c_ampcor.nLookAcross = a
    @property
    def downLooks(self):
        return self.c_ampcor.nLookDown
    @downLooks.setter
    def downLooks(self, int a):
        self.c_ampcor.nLookDown = a
    @property
    def oversamplingFactor(self):
        return self.c_ampcor.osampFact
    @oversamplingFactor.setter
    def oversamplingFactor(self, int a):
        self.c_ampcor.osampFact = a
    @property
    def zoomWindowSize(self):
        return self.c_ampcor.zoomWinSize
    @zoomWindowSize.setter
    def zoomWindowSize(self, int a):
        self.c_ampcor.zoomWinSize = a
    @property
    def acrossGrossOffset(self):
        return self.c_ampcor.acrossGrossOff
    @acrossGrossOffset.setter
    def acrossGrossOffset(self, int a):
        self.c_ampcor.acrossGrossOff = a
    @property
    def downGrossOffset(self):
        return self.c_ampcor.downGrossOff
    @downGrossOffset.setter
    def downGrossOffset(self, int a):
        self.c_ampcor.downGrossOff = a
    @property
    def debugFlag(self):
        return self.c_ampcor.corr_debug
    @debugFlag.setter
    def debugFlag(self, bool a):
        self.c_ampcor.corr_debug = a
    @property
    def displayFlag(self):
        return self.c_ampcor.corr_display
    @displayFlag.setter
    def displayFlag(self, bool a):
        self.c_ampcor.corr_display = a
    @property
    def usr_enable_gpu(self):
        return self.c_ampcor.usr_enable_gpu
    @usr_enable_gpu.setter
    def usr_enable_gpu(self, bool a):
        self.c_ampcor.usr_enable_gpu = a
    @property
    def numElem(self):
        if self._engine == isce3_engine.ENGINE_LEGACY:
            return self.c_ampcor.numRowTable
        return self._numElemPy

    def runAmpcor(self):
        if self._engine == isce3_engine.ENGINE_LEGACY:
            self.c_ampcor.ampcor()
            return
        self._runAmpcorIsce3()

    def _runAmpcorIsce3(self):
        ref_name = isce3_engine.resolve_image_name(self._referenceImageName)
        sec_name = isce3_engine.resolve_image_name(self._secondaryImageName)
        if ref_name is None:
            raise ValueError(
                "engine={0} requires referenceImageName for isce3 adapter path.".format(self._engine)
            )
        if sec_name is None:
            raise ValueError(
                "engine={0} requires secondaryImageName for isce3 adapter path.".format(self._engine)
            )
        if self._imageLength1 <= 0 or self._imageLength2 <= 0:
            raise ValueError(
                "engine={0} requires positive imageLength1/imageLength2 for isce3 adapter path.".format(
                    self._engine
                )
            )

        if self.c_ampcor.colSkip <= 0 or self.c_ampcor.rowSkip <= 0:
            raise ValueError("skipSampleAcross/skipSampleDown must be > 0.")

        n_across = ((self.c_ampcor.lastCol - self.c_ampcor.firstCol) // self.c_ampcor.colSkip) + 1
        n_down = ((self.c_ampcor.lastRow - self.c_ampcor.firstRow) // self.c_ampcor.rowSkip) + 1
        if n_across <= 0 or n_down <= 0:
            raise ValueError("Invalid offset grid size for isce3 adapter path.")

        corr_stat = max(5, 2 * int(self.c_ampcor.zoomWinSize) + 5)
        results = isce3_engine.run_ampcor_engine(
            engine=self._engine,
            reference_path=ref_name,
            secondary_path=sec_name,
            reference_width=int(self.c_ampcor.imgWidths[0]),
            reference_length=int(self._imageLength1),
            secondary_width=int(self.c_ampcor.imgWidths[1]),
            secondary_length=int(self._imageLength2),
            first_sample_across=int(self.c_ampcor.firstCol),
            first_sample_down=int(self.c_ampcor.firstRow),
            skip_sample_across=int(self.c_ampcor.colSkip),
            skip_sample_down=int(self.c_ampcor.rowSkip),
            number_window_across=int(n_across),
            number_window_down=int(n_down),
            window_size_width=int(self.c_ampcor.refChipWidth),
            window_size_height=int(self.c_ampcor.refChipHeight),
            search_window_size_width=int(self.c_ampcor.schMarginX),
            search_window_size_height=int(self.c_ampcor.schMarginY),
            across_gross_offset=int(self.c_ampcor.acrossGrossOff),
            down_gross_offset=int(self.c_ampcor.downGrossOff),
            oversampling_factor=int(self.c_ampcor.osampFact),
            zoom_window_size=int(self.c_ampcor.zoomWinSize),
            corr_surface_oversampling_method=0,
            raw_data_oversampling_factor=1,
            corr_stat_window_size=int(corr_stat),
            deramp_method=1,
            deramp_axis=0,
            device_id=0,
            n_streams=(2 if self._engine == isce3_engine.ENGINE_ISCE3_GPU else 1),
            chunk_window_across=min(64, int(n_across)),
            chunk_window_down=1,
            mmap_size_gb=8,
            use_mmap=1,
            output_prefix=None,
            cleanup_workdir=True,
        )

        self._locationAcrossPy = results['location_across'].tolist()
        self._locationDownPy = results['location_down'].tolist()
        self._locationAcrossOffsetPy = results['location_across_offset'].tolist()
        self._locationDownOffsetPy = results['location_down_offset'].tolist()
        self._snrPy = results['snr'].tolist()
        self._cov1Py = results['cov1'].tolist()
        self._cov2Py = results['cov2'].tolist()
        self._cov3Py = results['cov3'].tolist()
        self._numElemPy = int(results['num_rows'])

    def getLocationAcrossAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._locationAcrossPy[idx]
        else:
            return self.c_ampcor.getLocationAcrossAt(idx)
    def getLocationDownAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._locationDownPy[idx]
        else:
            return self.c_ampcor.getLocationDownAt(idx)
    def getLocationAcrossOffsetAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._locationAcrossOffsetPy[idx]
        else:
            return self.c_ampcor.getLocationAcrossOffsetAt(idx)
    def getLocationDownOffsetAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._locationDownOffsetPy[idx]
        else:
            return self.c_ampcor.getLocationDownOffsetAt(idx)
    def getSNRAt(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._snrPy[idx]
        else:
            return self.c_ampcor.getSnrAt(idx)
    def getCov1At(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._cov1Py[idx]
        else:
            return self.c_ampcor.getCov1At(idx)
    def getCov2At(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._cov2Py[idx]
        else:
            return self.c_ampcor.getCov2At(idx)
    def getCov3At(self, int idx):
        if (idx >= self.numElem):
            print("Error: Invalid element number ("+str(self.numElem)+" elements available).")
        elif self._engine != isce3_engine.ENGINE_LEGACY:
            return self._cov3Py[idx]
        else:
            return self.c_ampcor.getCov3At(idx)
    
