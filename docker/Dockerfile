FROM ubuntu:22.04 AS builder

# Set an encoding to make things work smoothly.
ENV LANG=en_US.UTF-8
ENV TZ=Asia/Shanghai
ARG DEBIAN_FRONTEND=noninteractive
ARG USE_CN_MIRROR=true
ARG APT_MIRROR=mirrors.aliyun.com

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN if [ "${USE_CN_MIRROR}" = "true" ]; then \
      for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "$f" ] || continue; \
        sed -i "s|http://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|https://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|http://security.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|https://security.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|http://ports.ubuntu.com/ubuntu-ports|https://${APT_MIRROR}/ubuntu-ports|g" "$f"; \
        sed -i "s|https://ports.ubuntu.com/ubuntu-ports|https://${APT_MIRROR}/ubuntu-ports|g" "$f"; \
      done; \
    fi

RUN set -ex \
 && apt-get update \
 && apt-get install -y \
    build-essential \
    cmake \
    cython3 \
    gfortran \
    git \
    snaphu \
    aria2 \
    libhdf5-dev \
    libhdf4-alt-dev \
    hdf5-tools \
    hdf4-tools \
    libopencv-dev \
    libx11-dev \
    libmotif-dev \
    libfftw3-dev \
    libgdal-dev \
    libhdf4-alt-dev \
    libhdf5-dev \
    libxm4 \
    libopencv-dev \
    ninja-build \
    libmotif-dev \
    python3-gdal \
    python3-h5py \
    python3-numpy \
    python3-opencv \
    python3-pip \
    python3-scipy \
    python3-venv \
 && apt clean \
 && echo done

# copy repo
COPY . /opt/isce2/src/isce2

# build ISCE
RUN set -ex \
 && cd /opt/isce2/src/isce2 \
 && mkdir build && cd build \
 && cmake .. \
        -DPYTHON_MODULE_DIR="$(python3 -c 'import site; print(site.getsitepackages()[-1])')" \
        -DCMAKE_INSTALL_PREFIX=install \
 && make -j8 install \
 && cpack -G DEB \
 && cp isce*.deb /tmp/

# Build an isolated Python environment for fast-moving contrib tools so the
# system Python stack used by ISCE/MintPy remains stable.
RUN set -ex \
 && python3 -m venv /opt/venvs/isce-extra \
 && . /opt/venvs/isce-extra/bin/activate \
 && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir /opt/isce2/src/isce2/contrib/spurt \
 && python -m pip install --no-cache-dir /opt/isce2/src/isce2/contrib/snaphu-py \
 && python -m pip install --no-cache-dir \
    numba \
    opera-utils \
    pydantic \
    pyproj \
    ruamel_yaml \
    threadpoolctl \
    tqdm \
    tyro

FROM ubuntu:22.04

# Set an encoding to make things work smoothly.
ENV LANG=en_US.UTF-8
ENV TZ=US/Pacific
ARG DEBIAN_FRONTEND=noninteractive
ARG USE_CN_MIRROR=true
ARG APT_MIRROR=mirrors.aliyun.com
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN if [ "${USE_CN_MIRROR}" = "true" ]; then \
      for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "$f" ] || continue; \
        sed -i "s|http://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|https://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|http://security.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|https://security.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" "$f"; \
        sed -i "s|http://ports.ubuntu.com/ubuntu-ports|https://${APT_MIRROR}/ubuntu-ports|g" "$f"; \
        sed -i "s|https://ports.ubuntu.com/ubuntu-ports|https://${APT_MIRROR}/ubuntu-ports|g" "$f"; \
      done; \
    fi

RUN set -ex \
 && apt-get update \
 && apt-get install -y \
    aria2 \
    curl \
    csh \
    gdal-bin \
    gmt \
    gmt-dcw \
    gmt-gshhg-low \
    libfftw3-3 \
    libgeos-dev \
    libgdal30 \
    libhdf4-0 \
    libhdf5-103-1 \
    libopencv-core4.5d \
    libopencv-highgui4.5d \
    libopencv-imgproc4.5d \
    libproj-dev \
    proj-bin \
    proj-data \
    libxm4 \
    nano \
    python3-argcomplete \
    python3-cartopy \
    python3-cvxopt \
    python3-dask \
    python3-distributed \
    python3-gdal \
    python3-h5py \
    python3-joblib \
    python3-lxml \
    python3-matplotlib \
    python3-pip \
    python3-numpy \
    python3-pyproj \
    python3-pyresample \
    python3-requests \
    python3-skimage \
    python3-shapely \
    python3-scipy \
    python3-mpi4py \
    python3-tqdm \
    python3-gdal \
    python3-astropy \
 && apt autoclean \
 && echo done

# Preconfigure Earthdata credentials for tools that still require .netrc
# (e.g., legacy DataRetriever/SWBD paths).
RUN set -ex \
 && printf '%s\n' \
    'machine urs.earthdata.nasa.gov' \
    '  login dongyusen' \
    '  password DYS123#cug' \
    > /root/.netrc \
 && chmod 600 /root/.netrc

# install ISCE from DEB
COPY --from=builder /tmp/isce*.deb /tmp/isce2.deb
COPY --from=builder /opt/venvs/isce-extra /opt/venvs/isce-extra
COPY --from=builder /opt/isce2/src/isce2/MintPy /opt/MintPy
COPY --from=builder /opt/isce2/src/isce2/contrib/stack /opt/isce2/contrib/stack
COPY --from=builder /opt/isce2/src/isce2/contrib/timeseries /opt/isce2/contrib/timeseries
COPY --from=builder /opt/isce2/src/isce2/contrib/snaphu-py /opt/isce2/contrib/snaphu-py
COPY --from=builder /opt/isce2/src/isce2/contrib/spurt /opt/isce2/contrib/spurt
COPY docker/isce2_raster_convert.py /usr/local/bin/isce2-raster-convert
COPY docker/isce2_tops_postprocess.py /usr/local/bin/isce2-tops-postprocess

RUN dpkg -i /tmp/isce2.deb
RUN ln -s /usr/lib/python3.10/dist-packages/isce2 /usr/lib/python3.10/dist-packages/isce || true

RUN set -ex \
 && PYTHONPATH=/usr/lib/python3.10/dist-packages/isce2/components:/usr/lib/python3.10/dist-packages/isce2:/usr/lib/python3.10/dist-packages:${PYTHONPATH} python3 - <<'PY'
import importlib
from pathlib import Path
import isce

modules = [
    'isceobj.InsarProc.runFilter',
    'isceobj.StripmapProc.runFilter',
    'isceobj.IsceProc.runFilter',
    'isceobj.TopsProc.runFilter',
    'isceobj.StripmapProc.runNormalizeSecondarySampling',
]

for name in modules:
    module = importlib.import_module(name)
    if name.endswith('runNormalizeSecondarySampling'):
        required = ('runNormalizeSecondarySampling',)
    else:
        required = ('_validated_alpha', '_validate_interferogram', '_validate_output')
    for symbol in required:
        if not hasattr(module, symbol):
            raise RuntimeError('{0} is missing {1}'.format(name, symbol))

stack_norm = Path('/opt/isce2/contrib/stack/stripmapStack/normalizeSecondarySampling.py')
if not stack_norm.exists():
    raise RuntimeError('Missing stripmapStack normalization script: {0}'.format(stack_norm))

print('ISCE2 validation hooks and stripmapStack normalization script verified in container.')
PY

RUN set -ex \
 && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir --no-deps \
    dask-jobqueue \
    pykml \
    pysolid \
    rich \
    utm \
 && python3 -m pip install --no-cache-dir pyaps3

# Use MintPy source tree directly (Python path is configured below) to keep
# compatibility with Python 3.8 and avoid pyproject build-time constraints.
RUN set -ex \
 && printf '%s\n' '#!/usr/bin/env bash' 'exec python3 -m mintpy "$@"' > /usr/local/bin/mintpy \
 && chmod +x /usr/local/bin/mintpy

RUN set -ex \
 && printf '%s\n' '#!/usr/bin/env bash' 'exec /opt/venvs/isce-extra/bin/python "$@"' > /usr/local/bin/isce-extra-python \
 && chmod +x /usr/local/bin/isce-extra-python

RUN chmod +x /usr/local/bin/isce2-raster-convert

ENV MINTPY_HOME=/opt/MintPy
ENV ISCE_HOME=/usr/lib/python3.10/dist-packages/isce2
ENV ISCE_CONTRIB=/opt/isce2/contrib
ENV ISCE_STACK=$ISCE_CONTRIB/stack
ENV ISCE_TIMESERIES=$ISCE_CONTRIB/timeseries
ENV SNAPHU_PY_HOME=$ISCE_CONTRIB/snaphu-py
ENV SPURT_HOME=$ISCE_CONTRIB/spurt
ENV PATH=$PATH:$ISCE_HOME/applications:$ISCE_STACK/topsStack:$ISCE_STACK/stripmapStack:$ISCE_STACK/alosStack:$ISCE_TIMESERIES/prepStackToStaMPS/bin:$MINTPY_HOME/src/mintpy/cli:/opt/venvs/isce-extra/bin
ENV PYTHONPATH=/usr/lib/python3.10/dist-packages:$ISCE_HOME/components:$ISCE_HOME:$ISCE_STACK:$MINTPY_HOME/src
ENV ISCE_SNAPHU_NPROC=8
ENV ISCE_SNAPHU_NTILEROW=2
ENV ISCE_SNAPHU_NTILECOL=2
ENV ISCE_SNAPHU_ROWOVRLP=400
ENV ISCE_SNAPHU_COLOVRLP=400

RUN ln -s /usr/lib/python3.10/dist-packages/isce2 /usr/lib/python3.10/dist-packages/isce
