# Optional CPU geometry dependencies for articulated-hand contact checks.
ARG BASE_IMAGE=r2s2r-hamer:3a01849
FROM ${BASE_IMAGE}
USER root
RUN pip install --no-cache-dir 'embreex==4.4.0.post1' 'rtree>=1.3,<2'
RUN python -c "from trimesh.ray.ray_pyembree import RayMeshIntersector; import smplx"
ENTRYPOINT ["python"]
