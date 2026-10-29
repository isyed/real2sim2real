# Build recipe for the extracted backends. GPU acceptance testing is separate.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ARG SAM2_REF
ARG SAM3D_REF
ARG NVDIFFRAST_REF
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
ENV PYOPENGL_PLATFORM=egl OPENCV_IO_ENABLE_OPENEXR=1 PYTHONUNBUFFERED=1
RUN test -n "$SAM2_REF" && test -n "$SAM3D_REF" && test -n "$NVDIFFRAST_REF"
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg build-essential cmake ninja-build python3-dev libeigen3-dev \
    libboost-all-dev libgl1 libegl1 libgles2 libglib2.0-0 libxrender1 \
    libxext6 libsm6 libx11-6 libosmesa6-dev libspatialindex-dev libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install numpy==1.26.3 torchvision==0.23.0 setuptools wheel ninja pybind11 \
    scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio \
    imageio-ffmpeg open3d transformations warp-lang einops kornia pyrender av h5py \
    hdf5plugin videoio==0.3.0 scikit-image meshcat webdataset omegaconf pypng roma \
    seaborn openpyxl matplotlib tqdm pillow accelerate timm yacs smplx tensorboardx \
    pycocotools rtree xatlas pytorch-lightning hydra-core hydra-submitit-launcher \
    hydra-colorlog pyrootutils networkx==3.2.1 rich dill pandas jsonlines loguru \
    optree fvcore cython py-soma-x==0.2.1 fast_simplification gdown psutil \
    astor easydict lightning pyvista pymeshfix igraph gsplat spconv-cu121==2.3.8
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
RUN FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
RUN pip install --no-deps 'git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
RUN pip install 'git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030'
RUN pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}"
RUN pip install --no-deps "git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}" \
    "git+https://github.com/facebookresearch/sam-3d-objects.git@${SAM3D_REF}"
WORKDIR /opt/real2sim2real
COPY . .
RUN pip install -e . -e modules/reconstruction
ENV ENGINE_ROOT=/opt/real2sim2real/modules/reconstruction/src/r2s2r_engines
RUN cmake -S "$ENGINE_ROOT/foundation_pose/lib/FoundationPose/mycpp" \
    -B "$ENGINE_ROOT/foundation_pose/lib/FoundationPose/mycpp/build" \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build "$ENGINE_ROOT/foundation_pose/lib/FoundationPose/mycpp/build" -j4
RUN pip install --no-build-isolation -e "$ENGINE_ROOT/foundation_pose/lib/FoundationPose/bundlesdf/mycuda"
ENV LIDRA_SKIP_INIT=1
WORKDIR /work
ENTRYPOINT ["r2s2r"]
