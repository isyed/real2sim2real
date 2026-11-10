# Build with the upstream HaWoR checkout as context. No weights or videos COPY.
ARG BASE_IMAGE=r2s2r-hamer:3a01849
FROM ${BASE_IMAGE}
USER root
ARG HAWOR_COMMIT=2c3fa0c06e54a1b192c1ff692eaa89d526cd6317
COPY . /opt/hawor
RUN test "$(git -C /opt/hawor rev-parse HEAD)" = "$HAWOR_COMMIT"
RUN pip install --no-cache-dir "ultralytics==8.3.70" "opencv-python<4.12" "numpy<2"
# lib is the pinned upstream package name; only the isolated worker uses it.
RUN python -c "import sysconfig; from pathlib import Path; (Path(sysconfig.get_paths()['purelib']) / 'hawor_backend.pth').write_text('/opt/hawor\\n')"
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
RUN python -c "from lib.models.hawor import HAWOR; from ultralytics import YOLO; print('HaWoR CPU imports OK')"
ENTRYPOINT ["python"]
