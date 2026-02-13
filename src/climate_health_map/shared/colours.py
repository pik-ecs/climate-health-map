from colorsys import hsv_to_rgb


# HSV (hue: 0–360°, saturation: 0–100%, value: 0–100%; aka HSB: brightness)
# HSL (hue: 0–360°, saturation: 0–100%, lightness: 0–100%)
# RGB (red: 0–255, green: 0–255, blue: 0–255)
# hue: 0(360) is red, 120 is green, 240 is blue; 180° gives complement colours
# saturation: 0% is grey, 100% is full colour
# lightness: 0% is black, 50% is normal, 100% is white
# value: 0% is black, 100% is full color
#   in colorsys, *all* values are floats normalised to 0–1


def hsv_to_hex(v: tuple[float, float, float]) -> str:
    if any(vi > 1 for vi in v):
        v = (v[0] / 360, v[1] / 100, v[2] / 100)
    values = [f'{int(vi):02x}'[-2:] for vi in hsv_to_rgb(*v)]
    return '#' + ''.join(values)


def hex_to_rgb(v: str) -> tuple[float, float, float]:
    return int(v[1:3], 16) / 255, int(v[3:5], 16) / 255, int(v[5:7], 16) / 255


def rgb_to_hex(v: tuple[float, float, float]) -> str:
    scale = 255 if all(vi < 1 for vi in v) else 1  # if all values <1, assume the input is using the normalised 0–1 scale and extend the range to 0–255
    return '#' + ''.join([f'{int(vi * scale):02x}'[-2:] for vi in v])
