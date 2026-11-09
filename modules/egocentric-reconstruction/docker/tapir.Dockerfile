# Build with a separately downloaded upstream checkout; no weights or videos.
ARG BASE_IMAGE=r2s2r-hamer:3a01849
FROM ${BASE_IMAGE}
USER root
ARG TAPIR_COMMIT=f2f8888a803051ea42cd79c89cfca6c25dfeae77
COPY . /opt/tapir
RUN test "$(git -C /opt/tapir rev-parse HEAD)" = "$TAPIR_COMMIT"
RUN pip install --no-cache-dir 'einshape==1.0' 'dm-tree==0.1.9'
RUN python -c "import sysconfig; from pathlib import Path; (Path(sysconfig.get_paths()['purelib']) / 'tapir_backend.pth').write_text('/opt/tapir\n')"
RUN python -c "from tapnet.torch.tapir_model import TAPIR; print('TAPIR CPU imports OK')"
ENTRYPOINT ["python"]
