#!/usr/bin/env bash
set -euo pipefail

# Template for applications/optical2sar.py (image2sar)
# 1) SAFE zip mode
python3 /usr/lib/python3.10/dist-packages/isce2/applications/optical2sar.py \
  --optical /Work/image2sar/optical_utm.tif \
  --reference /Work/image2sar/S1A_xxx.SAFE.zip \
  --product-type slc \
  --swath iw2 \
  --pol vv \
  --out /Work/image2sar/optical_on_sar_iw2.tif

# 2) ISCE XML mode (with existing lat/lon lookups)
# python3 /usr/lib/python3.10/dist-packages/isce2/applications/optical2sar.py \
#   --optical /Work/image2sar/optical_utm.tif \
#   --reference /Work/image2sar/reference.xml \
#   --lat-rdr /Work/image2sar/geometry/lat.rdr \
#   --lon-rdr /Work/image2sar/geometry/lon.rdr \
#   --out /Work/image2sar/optical_on_sar.tif

# 3) ISCE XML mode (auto-generate lookups)
# python3 /usr/lib/python3.10/dist-packages/isce2/applications/optical2sar.py \
#   --optical /Work/image2sar/optical_utm.tif \
#   --reference /Work/image2sar/reference.xml \
#   --dem /Work/dem/demLat_N28_N31_Lon_E093_E097.dem.wgs84 \
#   --rdr-dir /Work/image2sar/geometry \
#   --out /Work/image2sar/optical_on_sar.tif
