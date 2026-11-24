"""What SAM 2 was asked: re-plan the mask views exactly as the pipeline did (scanned poses,
`masks.plan_views`), and check which REAL chairs each body's prompt box contains."""
import json, numpy as np
from PIL import Image, ImageDraw
from common import *
from litereality_agent.models.splat import rgbd
from litereality_agent.pipeline.splatroom import masks as M
from litereality_agent.room_ops.roomcall.model_info import body_name, movable_body_ids
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/stray_atrium_q600")
m, d = model_data(B / "physics/scene.xml")
bodies = {body_name(m, b): b for b in sorted(movable_body_ids(m))}
views = M.plan_views(frames, m, d, bodies)
print("views", len(views), "boxes", sum(len(v.boxes) for v in views.values()))
real = {"blue chair": np.array([0.576, 1.337, -0.70]), "teal chair": np.array([-1.018, 0.016, -0.68]), "dark chair": np.array([-0.707, 0.416, -0.67])}
stats = {}
shown = 0
for key, v in views.items():
    for name, box in v.boxes.items():
        if not name.startswith("Chair"): continue
        st = stats.setdefault(name, {"views": 0, **{k: 0 for k in real}})
        st["views"] += 1
        for k, c in real.items():
            u, vv, z = M._project(v.K, v.frame.c2w, c[None])
            if z[0] > 0.2 and box[0] <= u[0] <= box[2] and box[1] <= vv[0] <= box[3]:
                st[k] += 1
    if shown < 4 and "Chair1" in v.boxes:
        im = Image.open(v.frame.image).convert("RGB").resize(v.size); dr = ImageDraw.Draw(im)
        for name, box, col in [(n, b, {"Chair0": (255, 50, 50), "Chair1": (50, 255, 50), "Chair2": (60, 120, 255)}.get(n, (255, 200, 0))) for n, b in v.boxes.items()]:
            dr.rectangle(box, outline=col, width=4); dr.text((box[0] + 5, box[1] + 5), name + " prompt", fill=col)
        for k, c in real.items():
            u, vv, z = M._project(v.K, v.frame.c2w, c[None])
            if z[0] > 0.2: dr.ellipse([u[0] - 9, vv[0] - 9, u[0] + 9, vv[0] + 9], fill=(255, 0, 255)); dr.text((u[0] + 12, vv[0]), k, fill=(255, 0, 255))
        im.rotate(-90, expand=True).save(OUT / f"sam_prompt_view{key}.jpg", quality=85); shown += 1
for n, st in stats.items():
    print(n, st)
json.dump(stats, open(OUT / "sam_prompts.json", "w"), indent=1)
