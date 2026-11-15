"""Published active-fire detection rules, applied to our Sentinel-2 test photos for a fair comparison.

Followed line by line from the reference implementation published with Pereira et al. 2021 (Landsat-8
active fire dataset, github.com/pereira-gha/activefire, src/landsat/create_masks.py):
  - Schroeder et al. 2016, Remote Sensing of Environment 185 (Landsat-8 OLI active fire)
  - Kumar & Roy 2018, Int. J. Digital Earth 11(2) (GOLI: global Landsat-8 active fire)
  - Murphy et al. 2016, Remote Sensing of Environment 177 (HOTMAP hot-target detection)
and from the paper itself (Table 3):
  - Hu, Ban & Nascetti 2021, Int. J. Applied Earth Observation and Geoinformation 101 (AFD-S2, Sentinel-2)

Adaptations to Sentinel-2 (the same for every method, written down so they can be checked):
  1. Bands: Landsat-8 OLI -> Sentinel-2 MSI equivalents: B2 blue -> B2, B3 -> B3, B4 -> B4, B5 NIR (0.865 um)
     -> B8A (0.865 um), B6 SWIR1 -> B11, B7 SWIR2 -> B12. OLI B1 (coastal, 0.44 um) is not in our data; the
     nearest band, B2 (0.49 um), is used instead (only Schroeder uses B1: Eq 2 and the water test).
  2. Windows: the contextual windows were defined in 30 m pixels; they are scaled to the same size on the
     ground in 20 m pixels (Schroeder 61x61 -> 91x91; Kumar-Roy growing window 5..61 -> 7..91). The 3x3
     neighbourhoods (adjacent pixels) stay 3x3.
  3. Input: each method is run twice: on Sentinel-2 L1C top-of-atmosphere reflectance, the input it was
     designed for (src/fetch_l1c.py; there the real B01 replaces the B2 substitute, and Schroeder gets
     reflectance without the sun-angle correction, i.e. times cos(sun zenith), as in the reference code), and on L2A surface
     reflectance, the same input our models get.
  4. Saturation (Murphy Eq 3): Landsat's BQA saturation bits -> Sentinel-2 scene classification class 1
     (saturated or defective).
  5. Murphy Eq 3 in the reference code divides by eps where B5 > 0 (an inverted np.where); the paper's
     formula (rho6 / rho5 >= 2 and rho6 >= 0.5) is used here.
Input to every function: dict b with keys 1..7 (Landsat band numbers) holding (H, W) float reflectance.
"""
import numpy as np
from scipy import ndimage

SQ3 = np.ones((3, 3), bool)


def landsat_bands(x7: np.ndarray) -> dict:
    """Our 7 bands (B12, B11, B8A, B4, B3, B2, B8) -> Landsat-8 numbering used by the published equations."""
    b12, b11, b8a, b4, b3, b2, _ = (np.asarray(v, np.float64) for v in x7)
    return {1: b2, 2: b2, 3: b3, 4: b4, 5: b8a, 6: b11, 7: b12}


def _window_index(n: int, half: int):
    i = np.arange(n)
    return np.maximum(0, i - half), np.minimum(n, i + half + 1)


def box_stats(values: np.ndarray, keep: np.ndarray, half: int):
    """Per pixel, over the (2*half+1)^2 window cut at the image edge (as the reference code does):
    number of kept pixels, their mean, their population std (np.nanmean / np.nanstd), and whether any kept
    value is infinite (the reference code's mean is then inf and its test fails)."""
    H, W = values.shape
    fin = keep & np.isfinite(values)
    v = np.where(fin, values, 0.0)
    t, b = _window_index(H, half)
    l, r = _window_index(W, half)

    def wsum(a):
        S = np.zeros((H + 1, W + 1))
        S[1:, 1:] = np.cumsum(np.cumsum(a, 0), 1)
        return S[b][:, r] - S[t][:, r] - S[b][:, l] + S[t][:, l]

    n = wsum(keep.astype(np.float64))
    n_fin = wsum(fin.astype(np.float64))
    s1, s2 = wsum(v), wsum(v * v)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n_fin
        std = np.sqrt(np.maximum(s2 / n_fin - mean * mean, 0.0))
    return n, mean, std, (n - n_fin) > 0.5


def window_area(H: int, W: int, half: int) -> np.ndarray:
    t, b = _window_index(H, half)
    l, r = _window_index(W, half)
    return (b - t)[:, None] * (r - l)[None, :]


# ------------------------------------------------------------------ Schroeder et al. 2016
def schroeder(b: dict) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r75 = b[7] / b[5]
        diff75 = b[7] - b[5]
        p6 = np.where(b[6] == 0, np.finfo(float).eps, b[6])
        eq6 = b[7] / p6 > 1.6
    unamb = ((b[7] > 0.5) & (r75 > 2.5) & (diff75 > 0.3)) | \
            ((b[6] > 0.8) & (b[1] < 0.2) & ((b[5] > 0.4) | (b[7] < 0.1)))              # Eq 1, Eq 2
    potential = (r75 > 1.8) & (diff75 > 0.17) & eq6                                      # Eq 3 and Eq 6
    water = (b[4] > b[5]) & (b[5] > b[6]) & (b[6] > b[7]) & (b[1] - b[7] < 0.2) & \
            ((b[3] > b[2]) | ((b[1] > b[2]) & (b[2] > b[3]) & (b[3] > b[4])))            # Eq 7, 8, 9
    if potential.any():                                                                  # Eq 4 and 5
        keep = ~((b[7] <= 0) | unamb | water)
        half = 45                                                                        # 61 px @30 m -> 91 px @20 m
        n, m75, s75, inf75 = box_stats(r75, keep, half)
        _, m7, s7, _ = box_stats(b[7], keep, half)
        with np.errstate(invalid="ignore"):
            eq4 = r75 > m75 + np.maximum(3 * s75, 0.8)
            eq5 = b[7] > m7 + np.maximum(3 * s7, 0.08)
        potential &= eq4 & eq5 & (n > 0) & ~inf75
    return (unamb | potential) & ~water


# ------------------------------------------------------------------ Kumar & Roy 2018 (GOLI)
def kumar_roy(b: dict) -> np.ndarray:
    valid = ndimage.binary_erosion(b[7] > 0, SQ3, border_value=1)                        # cv2.erode keeps edges
    unamb = (b[4] <= 0.53 * b[7] - 0.214) & valid                                        # Eq 12
    if unamb.any():                                                                      # Eq 13
        unamb |= ndimage.binary_dilation(unamb, SQ3) & (b[4] <= 0.35 * b[6] - 0.044)
        unamb &= valid
    potential = ((b[4] <= 0.53 * b[7] - 0.125) | (b[6] <= 1.08 * b[7] - 0.048)) & valid  # Eq 14, Eq 15
    water = (b[2] > b[3]) & (b[3] > b[4]) & (b[4] > b[5])                                # Eq 16
    if potential.any():                                                                  # Eq 8 and 9
        H, W = b[7].shape
        keep = ~(unamb | potential | water | ~valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            r75 = b[7] / b[5]
        decided = np.zeros((H, W), bool)
        passed = np.zeros((H, W), bool)
        for size in range(7, 92, 2):                                                     # 5..61 @30 m -> 7..91 @20 m
            half = size // 2
            n, m75, s75, inf75 = box_stats(r75, keep, half)
            _, m7, s7, _ = box_stats(b[7], keep, half)
            use = potential & ~decided & (n >= 0.25 * window_area(H, W, half))
            with np.errstate(invalid="ignore"):
                ok = (r75 > m75 + np.maximum(3 * s75, 0.8)) & (b[7] > m7 + np.maximum(3 * s7, 0.08)) & ~inf75
            passed |= use & ok
            decided |= use
            if not (potential & ~decided).any():
                break
        potential = passed                                                               # never tested -> not fire
    return (unamb | potential) & ~water


# ------------------------------------------------------------------ Murphy et al. 2016
def murphy(b: dict, saturated: np.ndarray) -> np.ndarray:
    p5 = np.where(b[5] == 0, np.finfo(float).eps, b[5])
    p6 = np.where(b[6] == 0, np.finfo(float).eps, b[6])
    unamb = (b[7] >= 0.15) & (b[7] / p6 >= 1.4) & (b[7] / p5 >= 1.4)                     # Eq 2
    if not unamb.any():
        return unamb
    near = ndimage.binary_dilation(unamb, SQ3)
    potential = near & (((b[6] / p5 >= 2.0) & (b[6] >= 0.5)) | saturated)                # Eq 3 (paper form)
    return unamb | potential


# ------------------------------------------------------------------ Hu, Ban & Nascetti 2021 (AFD-S2)
# Table 3: criteria per biome (RESOLVE / WWF biome names). Only these six biomes are covered by the paper.
HU_BIOMES = {
    "Tropical & Subtropical Moist Broadleaf Forests": dict(a=1.045, c=-0.071, ratio=True),
    "Tropical & Subtropical Dry Broadleaf Forests": dict(a=0.681, c=-0.052),
    "Tropical & Subtropical Grasslands, Savannas & Shrublands": dict(a=0.677, c=-0.052),
    "Mediterranean Forests, Woodlands & Scrub": dict(a=0.743, c=-0.068, med=True),
    "Temperate Conifer Forests": dict(a=0.504, c=-0.198),
    "Boreal Forests/Taiga": dict(a=0.727, c=-0.11),
}


def hu(b: dict, biome: str) -> np.ndarray | None:
    """None if the photo's biome isn't one the method was designed for."""
    k = HU_BIOMES.get(biome)
    if k is None:
        return None
    fire = b[4] <= k["a"] * b[7] + k["c"]                                                # C1 (Eq 2)
    if k.get("ratio"):
        with np.errstate(invalid="ignore", divide="ignore"):
            fire &= b[7] / b[6] >= 1                                                     # C2 (Eq 5)
    if k.get("med"):
        fire &= (b[7] >= 0.355) & ((b[6] >= 0.475) | (b[7] >= 1.0))                      # C2 (Eq 4), C3 (Eq 3 OR 6)
    return fire & (b[7] > 0)
