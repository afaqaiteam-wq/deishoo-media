"""Deishoo long-video thumbnail text, rendered in pixels (rules approved 2026-10-03).

Look = his approved Tut season-2 thumbnails: Cairo 900, white line then yellow, soft black shadow,
optional tatweel, small red season patch top-left (only for a season). New rules:
  - 2 lines, 2-4 words in total, a short Fusha question when it fits; the key line is white on a
    red box (Line(box=BOX_RED), use Block(gap=0.34)); yellow text is the older Tut look.
  - Lines are spaced on measured INK, so they can never touch (his note, twice).
  - Layer order: background -> text -> key element cutouts -> Mostafa cutout -> red marks -> patch,
    so no face or key object is ever covered (Tut's statue was covered once).
  - qa() reports how much of the text sits under a cutout, the gap between lines, and edge margins.

Arabic is shaped by Pillow's raqm layout (check: PIL.features.check('raqm') must be True).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H = 1920, 1080
WHITE = (255, 255, 255)
YELLOW = (255, 224, 0)  # #FFE000, the Tut s2 yellow
RED = (232, 16, 46)  # #E8102E, marks
PATCH_RED = (200, 16, 46)
BOX_RED = (225, 25, 35)  # key-word box (Dhruv-style), chosen over yellow on 2026-10-03
TATWEEL = "ـ"
FONT = str(Path(__file__).with_name("fonts") / "Cairo[slnt,wght].ttf")
_KW = dict(direction="rtl", language="ar")


def cairo(size: float, wght: int = 900) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(FONT, int(round(size)), layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([wght, 0])  # axes: Weight, Slant
    return f


# ---------------------------------------------------------------- text lines
@dataclass
class Line:
    """runs in reading order (first run = rightmost on screen): [(text, rgb), ...].
    A '|' inside a run marks where tatweel may be added (stretch)."""
    runs: list
    stretch: int = 0  # number of tatweel glyphs inserted at '|'
    scale: float = 1.0  # size relative to the block size
    box: tuple | None = None  # rgb: white text on a rounded box of this colour (the channel's red box)

    def text_runs(self):
        return [(t.replace("|", TATWEEL * self.stretch), c) for t, c in self.runs]


SS = 3  # supersampling: glyph joins are closed at 3x, then reduced, so no hairline seams show


def render_line(line: Line, size: float, wght: int = 900):
    """RGBA of the line cropped to its ink, plus the stretch (tatweel bar) x-range inside the crop.
    Drawn at 3x; a 3x3 closing fills the hairline seams FreeType leaves where joined glyphs meet."""
    f = cairo(size * line.scale * SS, wght)
    runs = line.text_runs()
    text = " ".join(t for t, _ in runs)
    adv = f.getlength(text, **_KW)
    pad = int(f.size)
    cw, ch = int(adv + 2 * pad), int(f.size * 2.8)
    cw += (-cw) % SS
    ch += (-ch) % SS
    base = int(f.size * 1.6)
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).text((pad, base), text, font=f, fill=255, anchor="ls", **_KW)
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    # colour each run: visual order is right to left, split colours in the middle of each space
    col = Image.new("RGB", (cw, ch), runs[0][1])
    d = ImageDraw.Draw(col)
    space = f.getlength(" ", **_KW)
    x_right = pad + adv
    bar = None
    for i, (t, c) in enumerate(runs):
        a = f.getlength(t, **_KW)
        x0 = x_right - a
        left = x0 - space / 2 if i < len(runs) - 1 else 0
        right = x_right + space / 2 if i > 0 else cw
        d.rectangle([left, 0, right, ch], fill=c)
        raw = line.runs[i][0]
        if "|" in raw and line.stretch:
            pre, post = raw.split("|", 1)
            # width of the part before the stretch, joined as it is drawn
            w_pre = f.getlength(pre + TATWEEL, **_KW) - f.getlength(TATWEEL, **_KW)
            w_bar = f.getlength(TATWEEL * line.stretch, **_KW)
            bar = ((x_right - w_pre - w_bar) / SS, (x_right - w_pre) / SS)
        x_right = x0 - space
    mask = mask.reduce(SS)
    col = col.reduce(SS)
    rgba = col.convert("RGBA")
    rgba.putalpha(mask)
    bb = mask.getbbox()
    out = rgba.crop(bb)
    if bar:
        bar = (bar[0] - bb[0], bar[1] - bb[0])
    return out, bar


def fit_stretch(line: Line, size: float, target_w: float, max_n: int = 12):
    """Pick the tatweel count that brings the line closest to target_w (never wider)."""
    if not any("|" in t for t, _ in line.runs):
        return line
    best = 0
    for n in range(0, max_n + 1):
        line.stretch = n
        w = render_line(line, size)[0].width
        if w <= target_w:
            best = n
        else:
            break
    line.stretch = best
    return line


@dataclass
class Block:
    lines: list
    size: float = 160
    gap: float = 0.22  # visible gap between lines, as a fraction of size, measured on ink
    align: str = "center"  # center | right | left
    balance: bool = True  # stretch shorter lines toward the widest one
    images: list = field(default_factory=list)
    bars: list = field(default_factory=list)
    offsets: list = field(default_factory=list)

    def build(self):
        if self.balance and len(self.lines) > 1:
            for ln in self.lines:
                ln.stretch = 0
            widths = [render_line(ln, self.size)[0].width for ln in self.lines]
            target = max(widths)
            for ln, w in zip(self.lines, widths):
                if w < target * 0.97:
                    fit_stretch(ln, self.size, target)
        rendered = [render_line(ln, self.size) for ln in self.lines]
        self.images = [r[0] for r in rendered]
        self.bars = [r[1] for r in rendered]
        g = int(round(self.gap * self.size))
        bw = max(im.width for im in self.images)
        bh = sum(im.height for im in self.images) + g * (len(self.images) - 1)
        canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        y = 0
        self.offsets = []
        for im in self.images:
            x = {"center": (bw - im.width) // 2, "right": bw - im.width, "left": 0}[self.align]
            canvas.alpha_composite(im, (x, y))
            self.offsets.append((x, y))
            y += im.height + g
        self.gap_px = g
        return canvas


@dataclass
class Row(Block):
    """Lines side by side on one row, first line on the right (reading order). Used when the free area
    is a wide, short band (e.g. above a helicopter): one line with its key part on the red box."""
    hgap: float = 0.45  # gap between the parts as a fraction of size (the box padding needs > 0.16)

    def build(self):
        rendered = [render_line(ln, self.size) for ln in self.lines]
        self.images = [r[0] for r in rendered]
        self.bars = [r[1] for r in rendered]
        g = int(round(self.hgap * self.size))
        bw = sum(im.width for im in self.images) + g * (len(self.images) - 1)
        bh = max(im.height for im in self.images)
        canvas = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        x = bw
        self.offsets = []
        for im in self.images:
            x -= im.width
            y = bh - im.height  # bottom-aligned: the shapes sit on one baseline area
            canvas.alpha_composite(im, (x, y))
            self.offsets.append((x, y))
            x -= g
        self.gap_px = g
        return canvas


def autosize(block: Block, max_w: float, max_h: float, lo=60, hi=320):
    """Largest size whose block fits max_w x max_h."""
    best = lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        block.size = mid
        im = block.build()
        if im.width <= max_w and im.height <= max_h:
            lo = best = mid
        else:
            hi = mid
    block.size = best
    return block.build()


# ---------------------------------------------------------------- effects
def shadowed(text: Image.Image, k: float = 1.0, stroke: float = 0.0):
    """Text with the Tut shadow (0 8px 18px .9 + 0 0 40px .55), scaled by k; optional dark outline."""
    pad = int(90 * k)
    a = text.getchannel("A")
    A = Image.new("L", (text.width + 2 * pad, text.height + 2 * pad), 0)
    A.paste(a, (pad, pad))
    drop = ImageChops.offset(A.filter(ImageFilter.GaussianBlur(9 * k)), 0, int(8 * k)).point(lambda v: int(v * 0.9))
    glow = A.filter(ImageFilter.GaussianBlur(20 * k)).point(lambda v: int(v * 0.55))
    sh = ImageChops.lighter(drop, glow)
    out = Image.new("RGBA", A.size, (0, 0, 0, 0))
    black = Image.new("RGBA", A.size, (0, 0, 0, 255))
    out = Image.composite(black, out, sh)
    if stroke > 0:
        r = max(1, int(round(stroke * k)))
        st = A.filter(ImageFilter.MaxFilter(2 * r + 1)).filter(ImageFilter.GaussianBlur(0.6 * k))
        st = st.point(lambda v: int(v * 0.7))
        out.alpha_composite(Image.composite(black, Image.new("RGBA", A.size, (0, 0, 0, 0)), st))
    out.alpha_composite(text, (pad, pad))
    return out, pad


def grade(im: Image.Image, contrast=1.06, color=1.1):
    rgb = im.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(color)
    if im.mode == "RGBA":
        rgb = rgb.convert("RGBA")
        rgb.putalpha(im.getchannel("A"))
    return rgb


def load(path, size=(W, H)):
    im = Image.open(path)
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im


def top_shade(strength=0.35, height=0.45):
    g = np.zeros((H, W), np.float32)
    hh = int(H * height)
    g[:hh] = (np.linspace(1, 0, hh) ** 1.6)[:, None] * strength
    return Image.fromarray((g * 255).astype(np.uint8), "L")


def mark(canvas: Image.Image, ellipse=None, arrow=None, width=14, color=RED):
    """Red ring (cx, cy, rx, ry, rot) and/or arrow [(x,y) bezier control points x4], drawn 2x and shadowed."""
    S = 2
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if ellipse:
        cx, cy, rx, ry, rot = ellipse
        pts = []
        for i in range(361):
            t = math.radians(i)
            x, y = rx * math.cos(t), ry * math.sin(t)
            r = math.radians(rot)
            pts.append(((cx + x * math.cos(r) - y * math.sin(r)) * S, (cy + x * math.sin(r) + y * math.cos(r)) * S))
        d.line(pts, fill=color, width=width * S, joint="curve")
    if arrow:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = arrow
        pts = []
        for i in range(101):
            t = i / 100
            x = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t ** 3 * x3
            y = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t ** 3 * y3
            pts.append((x * S, y * S))
        # stop the shaft short of the tip so the head is clean
        head = width * 3.2
        ex, ey = pts[-1]
        px, py = pts[-8]
        ang = math.atan2(ey - py, ex - px)
        shaft = [p for p in pts if math.hypot(p[0] - ex, p[1] - ey) > head * S * 0.6]
        d.line(shaft, fill=color, width=width * S, joint="curve")
        for p in (shaft[0],):
            d.ellipse([p[0] - width * S / 2, p[1] - width * S / 2, p[0] + width * S / 2, p[1] + width * S / 2], fill=color)
        hl = head * S
        tip = (ex + math.cos(ang) * hl * 0.25, ey + math.sin(ang) * hl * 0.25)
        left = (tip[0] - hl * math.cos(ang - 0.45), tip[1] - hl * math.sin(ang - 0.45))
        right = (tip[0] - hl * math.cos(ang + 0.45), tip[1] - hl * math.sin(ang + 0.45))
        d.polygon([tip, left, right], fill=color)
    layer = layer.resize((W, H), Image.LANCZOS)
    a = layer.getchannel("A")
    sh = ImageChops.offset(a.filter(ImageFilter.GaussianBlur(8)), 0, 6).point(lambda v: int(v * 0.85))
    canvas.alpha_composite(Image.composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), Image.new("RGBA", (W, H), (0, 0, 0, 0)), sh))
    canvas.alpha_composite(layer)


def patch(canvas: Image.Image, text="الموسم الثاني", x=28, y=28, size=30):
    """Small red season patch, top-left, like the Tut thumbnails. Only for a season.
    The box is sized on the text's measured ink, so dots under letters (ي) are never clipped."""
    f = cairo(size, 900)
    l, t, r, b = f.getbbox(text, anchor="ls", **_KW)
    px, py = 16, 9
    bw, bh = (r - l) + 2 * px, (b - t) + 2 * py
    box = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(box)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=8, fill=PATCH_RED + (235,))
    d.text((px - l, py - t), text, font=f, fill=WHITE, anchor="ls", **_KW)
    canvas.alpha_composite(box, (x, y))


# ---------------------------------------------------------------- compose + QA
@dataclass
class Thumb:
    bg: str
    block: Block
    box: tuple  # (x0, y0, x1, y1) where the text block must fit
    valign: str = "top"  # top | center | bottom inside the box
    cutouts: list = field(default_factory=list)  # RGBA paths drawn ABOVE the text (elements first, person last)
    shade: float = 0.0
    stroke: float = 0.0
    marks: list = field(default_factory=list)  # dicts for mark()
    season: str | None = None
    size: float | None = None  # fixed size instead of autosize
    shift: tuple = (0, 0)

    def render(self):
        bg = grade(load(self.bg)).convert("RGBA")
        if self.shade:
            bg.alpha_composite(Image.composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), Image.new("RGBA", (W, H), (0, 0, 0, 0)), top_shade(self.shade)))
        x0, y0, x1, y1 = self.box
        if self.size:
            self.block.size = self.size
            text = self.block.build()
        else:
            text = autosize(self.block, x1 - x0, y1 - y0)
        k = self.block.size / 160
        tx = {"center": x0 + (x1 - x0 - text.width) // 2, "right": x1 - text.width, "left": x0}[self.block.align] + self.shift[0]
        ty = {"top": y0, "center": y0 + (y1 - y0 - text.height) // 2, "bottom": y1 - text.height}[self.valign] + self.shift[1]
        out = bg.copy()
        text_alpha = Image.new("L", (W, H), 0)
        text_alpha.paste(text.getchannel("A"), (tx, ty))
        self.boxes = []
        if not any(ln.box for ln in self.block.lines):
            sh, pad = shadowed(text, k, self.stroke)
            out.alpha_composite(sh, (tx - pad, ty - pad))
        else:
            for ln, im, (ox, oy) in zip(self.block.lines, self.block.images, self.block.offsets):
                if not ln.box:
                    sh, pad = shadowed(im, k, self.stroke)
                    out.alpha_composite(sh, (tx + ox - pad, ty + oy - pad))
                    continue
                px, py = int(26 * k), int(12 * k)
                bx = (tx + ox - px, ty + oy - py, tx + ox + im.width + px, ty + oy + im.height + py)
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ImageDraw.Draw(layer).rounded_rectangle(bx, radius=int(10 * k), fill=tuple(ln.box) + (255,))
                soft = ImageChops.offset(layer.getchannel("A").filter(ImageFilter.GaussianBlur(14 * k)), 0, int(8 * k)).point(lambda v: int(v * 0.7))
                out.alpha_composite(Image.composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), Image.new("RGBA", (W, H), (0, 0, 0, 0)), soft))
                out.alpha_composite(layer)
                text_alpha.paste(255, tuple(max(0, v) for v in bx))
                out.alpha_composite(im, (tx + ox, ty + oy))
                self.boxes.append(bx)
        cover = Image.new("L", (W, H), 0)
        for c in self.cutouts:
            cut = grade(load(c))
            out.alpha_composite(cut)
            cover = ImageChops.lighter(cover, cut.getchannel("A"))
        for m in self.marks:
            mark(out, **m)
        if self.season:
            patch(out, self.season)
        self.text_pos = (tx, ty, text.width, text.height)
        self.qa_info = self._qa(text_alpha, cover)
        return out.convert("RGB")

    def _qa(self, text_alpha, cover):
        ta = np.array(text_alpha) > 128
        cv = np.array(cover) > 128
        hidden = (ta & cv).sum() / max(1, ta.sum())
        # hidden ink outside the tatweel bars (real letters) matters most
        letters = ta.copy()
        tx, ty = self.text_pos[:2]
        for (ox, oy), im, bar in zip(self.block.offsets, self.block.images, self.block.bars):
            if bar:
                letters[ty + oy: ty + oy + im.height, tx + ox + int(bar[0]) + 4: tx + ox + int(bar[1]) - 4] = False
        hidden_letters = (letters & cv).sum() / max(1, letters.sum())
        x, y, w, h = self.text_pos
        return {
            "size": self.block.size,
            "stretch": [ln.stretch for ln in self.block.lines],
            "gap_px": self.block.gap_px,
            "text_box": (x, y, x + w, y + h),
            "margins": (x, y, W - x - w),
            "hidden_ink_%": round(hidden * 100, 2),
            "hidden_letters_%": round(hidden_letters * 100, 2),
        }


def save(img: Image.Image, path, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=quality, optimize=True, progressive=True, subsampling=0)
    return path


def previews(img: Image.Image, stem, widths=(400, 210)):
    """Phone-size checks: 400 px = home feed on a phone, 210 px = 'up next' list."""
    out = []
    for w in widths:
        p = Path(f"{stem}-{w}.png")
        img.resize((w, round(w * 9 / 16)), Image.LANCZOS).save(p)
        out.append(p)
    return out


# ---------------------------------------------------------------- signature: tatweel behind his head
def obstacle(mask: np.ndarray, y0: int, y1: int, xmin: int, xmax: int):
    """x-range of the cutout inside rows y0..y1 and columns xmin..xmax (None if clear)."""
    band = mask[max(0, y0):min(H, y1), xmin:xmax].any(axis=0)
    xs = np.where(band)[0]
    if not len(xs):
        return None
    return xmin + int(xs[0]), xmin + int(xs[-1])


@dataclass
class Signature:
    """Each line is placed on its own so its tatweel bar runs BEHIND the head (approved Tut look):
    letters stay on both sides of the head, the bar hides behind it. Lines keep a measured ink gap."""
    lines: list
    size: float
    y_top: int
    head: str  # cutout whose head the bars run behind
    xmin: int
    xmax: int
    margin: int = 22
    gap: float = 0.22
    max_n: int = 40

    def place(self):
        mask = np.array(load(self.head).getchannel("A")) > 128
        g = int(round(self.gap * self.size))
        y = self.y_top
        placed = []
        for ln in self.lines:
            ln.stretch = 0
            im, bar = render_line(ln, self.size)
            ob = obstacle(mask, y, y + im.height, self.xmin, self.xmax)
            if ob is None:
                placed.append((im, None, None, y))
                y += im.height + g
                continue
            need = ob[1] - ob[0] + 2 * self.margin
            for n in range(1, self.max_n + 1):
                ln.stretch = n
                im, bar = render_line(ln, self.size)
                if bar and bar[1] - bar[0] >= need:
                    break
            x = int(ob[1] + self.margin - bar[1])  # bar's right end just past the head's right edge
            placed.append((im, bar, x, y))
            y += im.height + g
        self.placed = placed
        self.gap_px = g
        return placed


def render_signature(bg, sig: Signature, cutouts=(), season=None, shade=0.0, stroke=0.0, marks=()):
    out = grade(load(bg)).convert("RGBA")
    if shade:
        out.alpha_composite(Image.composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), Image.new("RGBA", (W, H), (0, 0, 0, 0)), top_shade(shade)))
    k = sig.size / 160
    text_alpha = Image.new("L", (W, H), 0)
    bars = []
    boxes = []
    for im, bar, x, y in sig.place():
        if x is None:
            x = (W - im.width) // 2
        sh, pad = shadowed(im, k, stroke)
        out.alpha_composite(sh, (x - pad, y - pad))
        text_alpha.paste(ImageChops.lighter(text_alpha.crop((x, y, x + im.width, y + im.height)), im.getchannel("A")), (x, y))
        boxes.append((x, y, x + im.width, y + im.height))
        if bar:
            bars.append((x + int(bar[0]) + 4, y, x + int(bar[1]) - 4, y + im.height))
    cover = Image.new("L", (W, H), 0)
    for c in cutouts:
        cut = grade(load(c))
        out.alpha_composite(cut)
        cover = ImageChops.lighter(cover, cut.getchannel("A"))
    for m in marks:
        mark(out, **m)
    if season:
        patch(out, season)
    ta = np.array(text_alpha) > 128
    cv = np.array(cover) > 128
    letters = ta.copy()
    for b in bars:
        letters[b[1]:b[3], b[0]:b[2]] = False
    info = {
        "size": sig.size,
        "stretch": [ln.stretch for ln in sig.lines],
        "gap_px": sig.gap_px,
        "boxes": boxes,
        "hidden_ink_%": round((ta & cv).sum() / max(1, ta.sum()) * 100, 2),
        "hidden_letters_%": round((letters & cv).sum() / max(1, letters.sum()) * 100, 2),
    }
    return out.convert("RGB"), info
