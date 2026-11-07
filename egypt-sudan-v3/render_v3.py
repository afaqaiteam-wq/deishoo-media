# Egypt-Sudan thumbnails v3 (Nov 7 2026). Run on Mostafa's PC:
#   python render_v3.py <kit_dir> <work_dir>
# work_dir holds «خام ChatGPT - ثامبنيل N.png»; outputs go to work_dir\out\
import sys, os
kit, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, kit)
from PIL import features
assert features.check('raqm'), "Pillow raqm missing: Arabic would not be shaped"
from thumbtext import *
out = os.path.join(work, "out"); os.makedirs(out, exist_ok=True)
raw = lambda n: os.path.join(work, f"خام ChatGPT - ثامبنيل {n}.png")
jobs = {
 "1": (raw(1), [Line([("مصر ضد", WHITE)]), Line([("السودان؟", WHITE)], scale=1.2, box=BOX_RED)], (1010, 48, 1872, 420)),
 "2": (raw(2), [Line([("حرب على", WHITE)]), Line([("الذهب؟", WHITE)], scale=1.25, box=BOX_RED)], (48, 48, 880, 420)),
 "3": (raw(3), [Line([("غارة على", WHITE)]), Line([("الحدود؟", WHITE)], scale=1.25, box=BOX_RED)], (48, 48, 840, 420)),
}
for k, (bg, lines, box) in jobs.items():
    t = Thumb(bg, Block(lines, balance=False, gap=0.34), box=box, shade=0.0)
    img = t.render()
    save(img, os.path.join(out, f"t{k}.jpg"))
    previews(img, os.path.join(out, f"t{k}"))
    print(k, t.qa_info)
