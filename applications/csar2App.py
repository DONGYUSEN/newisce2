#!/usr/bin/env python3

"""csar2App: China high-resolution stripmap InSAR application.

Design goals:
- Reuse StripmapProc modules to stay close to official ISCE2 path.
- Keep normalize_secondary_sampling enabled in the standard chain.
- Add optional radar-vs-DEM closure loop:
  rdrdem_offset -> rect_rgoffset to correct range.off geometry bias.
- Use existing Ampcor path (GPU preferred, CPU fallback, joint probe window
  selection) already integrated in runRefineSecondaryTiming.
- Provide modular satellite profiles for LT1 / GF3 / DJ1 and allow easy
  expansion.
"""

from __future__ import print_function

from isce import logging
from iscesys.Component.Application import Application

try:
    import stripmapApp as stripmap_base
except ImportError:
    from applications import stripmapApp as stripmap_base

try:
    from csar2_profiles import normalize_profile_name, sensor_for_profile
except ImportError:
    from applications.csar2_profiles import normalize_profile_name, sensor_for_profile

import isceobj.StripmapProc as StripmapProc
from isceobj.StripmapProc.Sensor import createSensor as createStripmapSensor

logger = logging.getLogger('isce.csar2')


CSAR2_PROFILE = Application.Parameter(
    'csar2Profile',
    public_name='csar2 profile',
    default='LT1',
    type=str,
    mandatory=False,
    doc='Satellite profile preset. Supported: LT1, GF3, DJ1.',
)

class CSAR2InSAR(stripmap_base.Insar):
    # Keep family compatible with existing stripmapApp.xml (<component name="insar">)
    # so current LT1/GF3/DJ1 configs can be reused directly.
    family = 'insar'
    parameter_list = stripmap_base._RoiBase.parameter_list + (
        CSAR2_PROFILE,
    )

    def __init__(self, family='', name='', cmdline=None):
        super().__init__(family=family if family else self.__class__.family, name=name, cmdline=cmdline)

    def Usage(self):
        print('Usages:')
        print('csar2App.py <input-file.xml>')
        print('csar2App.py --steps')
        print('csar2App.py --help')
        print('csar2App.py --help --steps')

    def _apply_csar2_profile_defaults(self):
        profile = normalize_profile_name(getattr(self, 'csar2Profile', 'LT1'), default='LT1')
        sensor = sensor_for_profile(profile)
        self.csar2Profile = profile

        if not getattr(self, 'sensorName', None):
            self.sensorName = sensor
        if not getattr(self, 'referenceSensorName', None):
            self.referenceSensorName = self.sensorName
        if not getattr(self, 'secondarySensorName', None):
            self.secondarySensorName = self.sensorName

        logger.info(
            'csar2 profile=%s resolved sensor defaults: sensorName=%s reference=%s secondary=%s',
            str(self.csar2Profile),
            str(self.sensorName),
            str(self.referenceSensorName),
            str(self.secondarySensorName),
        )

    def _configure(self):
        self._apply_csar2_profile_defaults()
        super()._configure()

        # Safety net: in csar2 profile mode, sensorName defaults may be applied
        # after facility construction. Ensure reference/secondary sensors exist.
        if getattr(self, 'reference', None) is None:
            self.reference = createStripmapSensor(
                getattr(self, 'sensorName', None),
                getattr(self, 'referenceSensorName', None),
                'reference',
            )
        if getattr(self, 'secondary', None) is None:
            self.secondary = createStripmapSensor(
                getattr(self, 'sensorName', None),
                getattr(self, 'secondarySensorName', None),
                'secondary',
            )

        # One switch controls closure execution and downstream rectified-range usage.
        self.enableRdrdemOffsetLoop = bool(getattr(self, 'enableRdrdemOffsetLoop', True))
        self.useRdrdemRectRangeOffset = bool(self.enableRdrdemOffsetLoop)

        logger.info(
            'csar2 switches: enableRdrdemOffsetLoop=%s useRdrdemRectRangeOffset=%s useGPU=%s useExternalCoregistration=%s',
            str(self.enableRdrdemOffsetLoop),
            str(self.useRdrdemRectRangeOffset),
            str(self.useGPU),
            str(self.useExternalCoregistration),
        )
        return None

    def _add_methods(self):
        super()._add_methods()
        self.runRdrDemOffset = StripmapProc.createRdrDemOffset(self)
        self.runRectRangeOffset = StripmapProc.createRectRangeOffset(self)
        return None

    def _steps(self):
        self.step('startup', func=self.startup,
                  doc='Print a helpful message and set the startTime of processing')

        self.step('preprocess', func=self.runPreprocessor,
                  doc='Preprocess reference/secondary data')

        self.step('cropraw', func=self.runCrop, args=(True,))
        self.step('formslc', func=self.runFormSLC)
        self.step('cropslc', func=self.runCrop, args=(False,))

        self.step('verifyDEM', func=self.verifyDEM)
        self.step('topo', func=self.runTopo)

        # Core normalization pipeline for high-resolution Chinese stripmap data.
        self.step('normalize_secondary_sampling', func=self.runNormalizeSecondarySampling)

        self.step('geo2rdr', func=self.runGeo2rdr)
        self.step('rdrdem_offset', func=self.runRdrDemOffset)
        self.step('rect_rgoffset', func=self.runRectRangeOffset)

        self.step('coarse_resample', func=self.runResampleSlc, args=('coarse',))
        self.step('misregistration', func=self.runRefineSecondaryTiming)
        self.step('refined_resample', func=self.runResampleSlc, args=('refined',))

        self.step('dense_offsets', func=self.runDenseOffsets)
        self.step('rubber_sheet_range', func=self.runRubbersheetRange)
        self.step('rubber_sheet_azimuth', func=self.runRubbersheetAzimuth)
        self.step('fine_resample', func=self.runResampleSlc, args=('fine',))

        self.step('split_range_spectrum', func=self.runSplitSpectrum)
        self.step('sub_band_resample', func=self.runResampleSubbandSlc, args=(True,))

        self.step('interferogram', func=self.runInterferogram)
        self.step('sub_band_interferogram', func=self.runInterferogram, args=('sub',))

        self.step('filter', func=self.runFilter, args=(self.filterStrength,))
        self.step('filter_low_band', func=self.runFilter, args=(self.filterStrength, 'low',))
        self.step('filter_high_band', func=self.runFilter, args=(self.filterStrength, 'high',))

        self.step('unwrap', func=self.runUnwrapper)
        self.step('unwrap_low_band', func=self.runUnwrapper, args=('low',))
        self.step('unwrap_high_band', func=self.runUnwrapper, args=('high',))

        self.step('ionosphere', func=self.runDispersive)
        self.step('geocode', func=self.runGeocode, args=(self.geocode_list, self.geocode_bbox))
        self.step('geocodeoffsets', func=self.runGeocode, args=(self.off_geocode_list, self.geocode_bbox, True))

        self.step('endup', func=self.endup)
        return None

    def main(self):
        self.timeStart = stripmap_base.time.time()
        self._insar.timeStart = self.timeStart
        self.help()

        self.runPreprocessor()
        self.runCrop(True)
        self.runFormSLC()
        self.runCrop(False)

        self.verifyDEM()
        self.runTopo()
        self.runNormalizeSecondarySampling()

        self.runGeo2rdr()
        self.runRdrDemOffset()
        self.runRectRangeOffset()

        self.runResampleSlc('coarse')
        self.runRefineSecondaryTiming()
        self.runResampleSlc('refined')

        self.runDenseOffsets()
        self.runRubbersheetAzimuth()
        self.runRubbersheetRange()
        self.runResampleSlc('fine')

        self.runSplitSpectrum()
        self.runResampleSubbandSlc(misreg=True)
        self.runInterferogram()
        self.runInterferogram(igramSpectrum='sub')

        self.runFilter(self.filterStrength)
        self.runFilter(self.filterStrength, igramSpectrum='low')
        self.runFilter(self.filterStrength, igramSpectrum='high')

        self.runUnwrapper()
        self.runUnwrapper(igramSpectrum='low')
        self.runUnwrapper(igramSpectrum='high')

        self.runDispersive()
        self.runGeocode(self.geocode_list, self.geocode_bbox)
        self.runGeocode(self.off_geocode_list, self.geocode_bbox, True)

        self.endup()
        return None


if __name__ == '__main__':
    app = CSAR2InSAR(name='csar2App')
    app.configure()
    status = app.run()
    raise SystemExit(status)
