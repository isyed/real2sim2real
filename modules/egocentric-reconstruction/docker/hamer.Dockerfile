# Reuse a CUDA 12.8 / PyTorch 2.8 reconstruction runtime on Blackwell GPUs.
# Override BASE_IMAGE with a compatible image containing the dependencies below.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG HAMER_COMMIT=3a01849f4148352e9260b69bf28b65d1671a4905
ARG INSTALL_DEPENDENCIES=1
USER root
RUN if [ "$INSTALL_DEPENDENCIES" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends git libgl1 libegl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir "numpy<2" scipy Pillow opencv-python-headless trimesh \
    pyrender==0.1.45 smplx==0.1.28 yacs==0.1.8 hydra-core omegaconf \
    pytorch-lightning==2.6.6 timm==1.0.30 einops webdataset braceexpand dill \
    joblib loguru scikit-image; fi
RUN git clone https://github.com/geopavlakos/hamer.git /opt/hamer && \
    cd /opt/hamer && git checkout ${HAMER_COMMIT}
# Chumpy is needed only to deserialize the separately supplied MANO asset.
RUN pip install --no-cache-dir --no-deps chumpy==0.70 --no-build-isolation
RUN pip install --no-cache-dir --no-deps -e /opt/hamer
ENV PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1
RUN python -c "from hamer.models import HAMER; from hamer.datasets.vitdet_dataset import ViTDetDataset; print('HaMeR CPU imports OK')"
# Inference mounts code, observations and model stores explicitly; no data COPY.
ENTRYPOINT ["python"]
