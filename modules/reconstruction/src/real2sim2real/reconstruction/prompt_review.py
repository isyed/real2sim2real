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


def review_first_frame(image_file, prompts_file, output, *, frame_count, source_video_sha256):
    from PIL import Image, ImageDraw

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError('Expected a positive source frame count')
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
            if frame == 0:
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
                if frame == 0:
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color if label else 'red')
        checked.append(dict(frame_index=frame, object_id=identity, kind=kind))
        if frame:
            later.add(frame)
    output.mkdir(parents=True, exist_ok=True)
    preview = output / 'first_frame_prompts.png'
    image.save(preview)
    result = dict(schema='real2sim2real.prompt-review/1', native_size=[width, height],
        frame_count=frame_count, source_video_sha256=source_video_sha256,
        first_rgb_sha256=digest(image_file), prompts_sha256=digest(prompts_file),
        preview_sha256=digest(preview), checked_prompts=checked, coordinates_in_bounds=True,
        semantics_verified=False, unpreviewed_prompt_frames=sorted(later),
        limitations=['In-bounds prompts may still identify the wrong person or object; inspect the RGB preview.',
            'Only frame-zero boxes and points are drawn here; later-frame and mask prompts require their own image review.'])
    write_json(output / 'prompt_review.json', result)
    return result
