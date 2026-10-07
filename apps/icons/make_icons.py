"""Draw the app icons: python apps/icons/make_icons.py (needs Pillow).

Without an icon, PyInstaller gives every build its own default -- the
Python snake on Windows, a blank app on macOS -- so the Hub Counter and
Watchtower looked alike and like a script. The finished files are
committed (HubCounter.icns/.ico/.png, Watchtower.*), so builds need no
Pillow; run this only to change the drawings. Colours match the pages'
the team's red and black: Hub Counter a red ball over a white hub on black,
Watchtower a black lighthouse on red.
"""
import math
import os
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
S = 4                       # supersampling: drawn at 4096, scaled to 1024
N = 1024 * S
RED = (239, 35, 60)
BLACK = (17, 17, 19)


def gradient(c0, c1):
    """A diagonal gradient, top left c0 to bottom right c1."""
    v = Image.linear_gradient("L").resize((N, N))
    g = ImageChops.add(v.point(lambda x: x // 2), v.rotate(90).point(lambda x: x // 2))
    return Image.composite(Image.new("RGB", (N, N), c1), Image.new("RGB", (N, N), c0), g)


def squircle_mask():
    """macOS's grid: an 824/1024 body, corners ~22% of it, transparent margin
    (Windows and Linux show the same shape)."""
    m = Image.new("L", (N, N), 0)
    pad = 100 * S
    ImageDraw.Draw(m).rounded_rectangle((pad, pad, N - pad, N - pad), radius=185 * S, fill=255)
    return m


def finish(art, c0, c1, name):
    body = gradient(c0, c1).convert("RGBA")
    # a soft top highlight, so the tile is not flat
    hi = Image.new("L", (N, N), 0)
    ImageDraw.Draw(hi).ellipse((-N // 3, -N * 2 // 3, N * 4 // 3, N // 2), fill=46)
    body = Image.composite(Image.new("RGBA", (N, N), (255, 255, 255, 255)), body,
                           hi.filter(ImageFilter.GaussianBlur(60 * S)))
    body.alpha_composite(art)
    mask = squircle_mask()
    shadow = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    sm = mask.filter(ImageFilter.GaussianBlur(18 * S)).point(lambda v: v * 0.45)
    shadow.putalpha(ImageChops.offset(sm, 0, 14 * S))
    out = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    out.alpha_composite(shadow)
    tile = body.copy()
    tile.putalpha(mask)
    out.alpha_composite(tile)
    img = out.resize((1024, 1024), Image.LANCZOS)
    img.save(os.path.join(HERE, f"{name}.png"))
    img.save(os.path.join(HERE, f"{name}.ico"),
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.save(os.path.join(HERE, f"{name}.icns"))
    print("wrote", name)


def p(x, y):
    return (x * S, y * S)


def hub_counter():
    art = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(art)
    # the ball's arc into the funnel: three fading dots
    for x, y, r in ((292, 330, 20), (360, 262, 28), (444, 220, 36)):
        d.ellipse((*p(x - r, y - r), *p(x + r, y + r)), fill=RED + (255,))
    # the hub: a funnel, translucent with a white rim
    funnel = [p(300, 470), p(724, 470), p(650, 800), p(374, 800)]
    d.polygon(funnel, fill=(255, 255, 255, 34))
    d.line(funnel + funnel[:2], fill=(255, 255, 255, 255), width=36 * S, joint="curve")
    for q in funnel:                          # round the corners
        r = 18 * S
        d.ellipse((q[0] - r, q[1] - r, q[0] + r, q[1] + r), fill=(255, 255, 255, 255))
    # mesh lines on the funnel
    for f in (0.33, 0.66):
        y = 470 + f * 330
        xl = 300 + f * 74
        d.line([p(xl + 30, y), p(1024 - xl - 30, y)], fill=(255, 255, 255, 120), width=14 * S)
    # the ball, about to drop in
    cx, cy, r = 540, 370, 78
    d.ellipse((*p(cx - r, cy - r), *p(cx + r, cy + r)), fill=RED + (255,))
    d.ellipse((*p(cx - r * 0.55, cy - r * 0.6), *p(cx - r * 0.05, cy - r * 0.15)),
              fill=(255, 150, 160, 220))
    finish(art, (52, 52, 58), (6, 6, 8), "HubCounter")


def watchtower():
    art = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(art)
    # the light's beams, behind the tower
    beams = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    bd = ImageDraw.Draw(beams)
    for side in (-1, 1):
        bd.polygon([p(512, 330), p(512 + side * 430, 230), p(512 + side * 430, 410)],
                   fill=(255, 225, 225, 150))
    # fade the beams out towards the edges
    fade = Image.linear_gradient("L").rotate(90).resize((N // 2, N))
    ramp = Image.new("L", (N, N), 0)
    ramp.paste(fade.transpose(Image.FLIP_LEFT_RIGHT), (0, 0))
    ramp.paste(fade, (N // 2, 0))
    ramp = ramp.point(lambda x: 255 - x)
    beams.putalpha(ImageChops.multiply(beams.getchannel("A"), ramp))
    art.alpha_composite(beams.filter(ImageFilter.GaussianBlur(5 * S)))
    white = BLACK + (255,)           # the tower's colour (black on red)
    # roof
    d.polygon([p(512, 175), p(625, 268), p(399, 268)], fill=white)
    # lamp room: a dark window with the lit lamp
    d.rounded_rectangle((*p(410, 268), *p(614, 392)), radius=14 * S, fill=white)
    d.rounded_rectangle((*p(440, 292), *p(584, 368)), radius=10 * S, fill=(255, 255, 255, 255))
    d.ellipse((*p(482, 300), *p(542, 360)), fill=RED + (255,))
    # gallery
    d.rounded_rectangle((*p(370, 392), *p(654, 430)), radius=12 * S, fill=white)
    # tapering tower
    d.polygon([p(430, 430), p(594, 430), p(640, 800), p(384, 800)], fill=white)
    # stripes and a door
    for y0, y1 in ((520, 560), (650, 690)):
        f0, f1 = (y0 - 430) / 370, (y1 - 430) / 370
        d.polygon([p(430 - 46 * f0, y0), p(594 + 46 * f0, y0),
                   p(594 + 46 * f1, y1), p(430 - 46 * f1, y1)], fill=(255, 255, 255, 255))
    d.rounded_rectangle((*p(486, 715), *p(538, 800)), radius=24 * S, fill=RED + (255,))
    # ground line
    d.rounded_rectangle((*p(300, 790), *p(724, 822)), radius=16 * S, fill=white)
    finish(art, (244, 63, 78), (150, 10, 24), "Watchtower")


if __name__ == "__main__":
    which = sys.argv[1:] or ["HubCounter", "Watchtower"]
    if "HubCounter" in which:
        hub_counter()
    if "Watchtower" in which:
        watchtower()
