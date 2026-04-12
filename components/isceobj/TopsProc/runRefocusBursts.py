"""
Optional burst SLC refocusing step using time-domain backprojection.

Refocuses each burst SLC to a unified zero-Doppler geometry,
which can improve coregistration and reduce TOPS artifacts.
"""

import logging
import os
import numpy as np

logger = logging.getLogger("isce.tops.runRefocusBursts")


def runRefocusBursts(self):
    """
    Iterate over all swaths and bursts, refocus each burst SLC
    using backprojection, and overwrite the burst SLC files.
    """
    from isce3_backproject.adapters.isce2_adapter import (
        orbit_from_isce2,
        lookside_from_isce2,
    )
    from isce3_backproject.adapters.tops_adapter import refocus_burst
    from isce3_backproject.adapters.dem_adapter import dem_from_file
    from isce3_backproject.backproject import DEMInterpolator

    dem_path = getattr(self._insar, "demFilename", None)
    if dem_path and os.path.exists(dem_path):
        dem = dem_from_file(dem_path)
    else:
        dem = DEMInterpolator(0.0)

    for scene_key in ["reference", "secondary"]:
        product_attr = f"{scene_key}SlcProduct"
        product_xml = getattr(self._insar, product_attr, None)
        if product_xml is None:
            logger.info("No %s SLC product, skipping refocusing.", scene_key)
            continue

        product = self._insar.loadProduct(product_xml)

        for swath_num, swath in enumerate(
            product.bursts if hasattr(product, "bursts") else [product]
        ):
            if not hasattr(swath, "__iter__"):
                bursts = [swath]
            else:
                bursts = swath

            orbit_isce3 = orbit_from_isce2(bursts[0].orbit)
            look_side = lookside_from_isce2(
                bursts[0].instrument.platform.pointingDirection
                if hasattr(bursts[0], "instrument")
                else -1
            )

            for burst_idx, burst in enumerate(bursts):
                logger.info(
                    "Refocusing %s swath %d burst %d...",
                    scene_key,
                    swath_num + 1,
                    burst_idx + 1,
                )

                refocused = refocus_burst(burst, orbit_isce3, look_side, dem=dem)

                outfile = burst.image.filename
                refocused.astype(np.complex64).tofile(outfile)
                logger.info("  Written: %s", outfile)

    return None
