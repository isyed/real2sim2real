"""Validate pixel coordinates and expose prompts on the actual source RGB.

Bounds checks cannot determine whether a prompt identifies the intended subject.
The preview is inspection evidence, never an automatic semantic approval.
"""
from pathlib import Path
import json
import math

from real2sim2real.artifacts import digest, write_json


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def review_frame(image_file, prompts_file, output, *, frame_count, source_video_sha256, frame_index=0):
    from PIL import Image, ImageDraw

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError('Expected a positive source frame count')
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index < frame_count:
        raise ValueError('Preview frame index is outside the source timeline')
    if (not isinstance(source_video_sha256, str) or len(source_video_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_video_sha256)):
        raise ValueError('Expected the source video SHA-256')
    image_file, prompts_file, output = map(Path, (image_file, prompts_file, output))
    with Image.open(image_file) as source:
        image = source.convert('RGB')
    width, height = image.size
    document = json.loads(prompts_file.read_text())
    prompts = document.get('prompts')
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('Expected nonempty segmentation prompts')
    draw = ImageDraw.Draw(image)
    checked = []
    later = set()
    for prompt in prompts:
        frame, identity = prompt.get('frame_index'), prompt.get('object_id')
        if (isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frame_count
                or isinstance(identity, bool) or not isinstance(identity, int) or identity < 0):
            raise ValueError('Prompt frame or object identity is invalid')
        kinds = [key for key in ('box', 'points', 'mask_path') if prompt.get(key)]
        if len(kinds) != 1:
            raise ValueError('Each prompt needs exactly one box, point set or mask')
        kind = kinds[0]
        color = ['lime', 'orange', 'cyan', 'magenta'][identity % 4]
        if kind == 'box':
            box = prompt['box']
            values = [box.get(k) for k in ('x0', 'y0', 'x1', 'y1')]
            if not all(_number(x) for x in values):
                raise ValueError('Box coordinates must be finite native pixels')
            x0, y0, x1, y1 = values
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('Prompt box is outside native RGB or has zero extent')
            if frame == frame_index:
                draw.rectangle(values, outline=color, width=max(2, width // 300))
                draw.text((x0, max(0, y0 - 16)), f'ID {identity}', fill=color, stroke_width=1)
        elif kind == 'points':
            points, labels = prompt['points'], prompt.get('point_labels')
            if not isinstance(points, list) or not isinstance(labels, list) or len(points) != len(labels):
                raise ValueError('Points require one foreground/background label each')
            for point, label in zip(points, labels):
                x, y = point.get('x'), point.get('y')
                if not (_number(x) and _number(y) and 0 <= x < width and 0 <= y < height and label in (0, 1)):
                    raise ValueError('Prompt point or label is outside its native contract')
                if frame == frame_index:
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color if label else 'red')
        checked.append(dict(frame_index=frame, object_id=identity, kind=kind))
        if frame != frame_index:
            later.add(frame)
    output.mkdir(parents=True, exist_ok=True)
    preview = output / ('first_frame_prompts.png' if frame_index == 0 else f'frame_{frame_index:06d}_prompts.png')
    image.save(preview)
    result = dict(schema='real2sim2real.prompt-review/1', native_size=[width, height],
        frame_count=frame_count, source_video_sha256=source_video_sha256,
        rgb_frame_sha256=digest(image_file), rgb_frame_index=frame_index, prompts_sha256=digest(prompts_file),
        preview_sha256=digest(preview), checked_prompts=checked, coordinates_in_bounds=True,
        semantics_verified=False, unpreviewed_prompt_frames=sorted(later),
        limitations=['In-bounds prompts may still identify the wrong person or object; inspect the RGB preview.',
            'Only this source frame\'s boxes and points are drawn; other frames and mask-file prompts require separate review.'])
    if frame_index == 0:
        result['first_rgb_sha256'] = digest(image_file)
    write_json(output / 'prompt_review.json', result)
    return result


def review_first_frame(image_file, prompts_file, output, *, frame_count, source_video_sha256):
    """Backward-compatible frame-zero inspection entry point."""
    return review_frame(image_file, prompts_file, output, frame_count=frame_count,
                        source_video_sha256=source_video_sha256)


def review_prompt_frames(images, prompts_file, output, *, frame_count, source_video_sha256):
    """Inspect every annotated source frame using explicitly indexed RGB files.

    Bounds/coverage are checked, but semantic identity remains a visual decision.
    The caller must supply actual source frames, not recropped/renumbered images.
    """
    document = json.loads(Path(prompts_file).read_text())
    required = {p['frame_index'] for p in document.get('prompts', [])}
    if not required or not required.issubset(images):
        raise ValueError('Supply actual RGB images for every prompt frame')
    output = Path(output)
    rows = []
    for frame in sorted(required):
        folder = output / f'frame_{frame:06d}'
        result = review_frame(images[frame], prompts_file, folder, frame_count=frame_count,
                              source_video_sha256=source_video_sha256, frame_index=frame)
        rows.append(dict(frame_index=frame, rgb_sha256=result['rgb_frame_sha256'],
                         preview_sha256=result['preview_sha256'],
                         report_sha256=digest(folder/'prompt_review.json')))
    result = dict(schema='real2sim2real.multi-frame-prompt-review/1', prompt_frames=sorted(required),
                  all_prompt_frames_previewed=True, frames=rows, semantics_verified=False,
                  prompts_sha256=digest(prompts_file), source_video_sha256=source_video_sha256)
    write_json(output/'prompt_frames_review.json', result)
    return result
