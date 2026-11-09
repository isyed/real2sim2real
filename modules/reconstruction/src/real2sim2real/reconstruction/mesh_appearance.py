"""Keep genuine appearance when adapting meshes for FoundationPose."""
import numpy as np


def foundationpose_visual(mesh):
    """Convert texture visuals only; never erase existing vertex/face colours.

    ColorVisuals has no to_color method. Treating that AttributeError as missing
    appearance silently replaced real painted features with uniform grey.
    Actual conversion failures are now explicit rather than hidden fallbacks.
    """
    import trimesh
    kind=mesh.visual.kind
    if kind=="texture":
        try:
            visual=mesh.visual.to_color()
        except Exception as error:
            raise ValueError("Cannot convert source texture; refusing to discard RGB cues") from error
        mesh.visual=visual
    elif kind not in ("vertex","face"):
        colors=np.tile(np.array([[180,180,180,255]],np.uint8),(len(mesh.vertices),1))
        mesh.visual=trimesh.visual.color.ColorVisuals(mesh=mesh,vertex_colors=colors)
        mesh.metadata["tracking_appearance"]="missing_source_visual_default_grey"
        return mesh
    colors=np.asarray(mesh.visual.vertex_colors)
    if colors.ndim!=2 or len(colors)!=len(mesh.vertices) or colors.shape[1] not in (3,4) or not np.isfinite(colors).all():
        raise ValueError("Source mesh colour coverage is invalid")
    mesh.metadata["tracking_appearance"]="preserved_"+str(kind)
    return mesh
