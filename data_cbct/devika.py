import math
import cv2
import numpy as np

SCALE_MM_TOOTH = 10
SCALE_MM_PULP  = 5
CUBE_TOOTH     = 22.358
CUBE_PULP      = 12.867
CUBE_PIXEL_LEN = 203
AGE_TRUE_LABEL = 23

def main():
    devika_pulp           = cv2.imread("kari/21-30/females/Devika@23/pulp.bmp",           cv2.IMREAD_GRAYSCALE)
    devika_tooth          = cv2.imread("kari/21-30/females/Devika@23/tooth.bmp",          cv2.IMREAD_GRAYSCALE)
    devika_pulp_original  = cv2.imread("kari/21-30/females/Devika@23/pulp_original.bmp",  cv2.IMREAD_GRAYSCALE)
    devika_tooth_original = cv2.imread("kari/21-30/females/Devika@23/tooth_original.bmp", cv2.IMREAD_GRAYSCALE)
    
    devika_pulp_scale_px  = detect_pixels(devika_pulp_original)
    devika_tooth_scale_px = detect_pixels(devika_tooth_original)

    mmpx_pulp  = SCALE_MM_PULP  / devika_pulp_scale_px
    mmpx_tooth = SCALE_MM_TOOTH / devika_tooth_scale_px

    devika_tooth_white_pixels = cv2.countNonZero(devika_tooth)
    devika_pulp_white_pixels  = cv2.countNonZero(devika_pulp)
    
    devika_pulp_to_tooth_pixel_ratio  = devika_pulp_white_pixels / devika_tooth_white_pixels
    devika_pulp_to_tooth_volume_ratio = (((math.cbrt(CUBE_PULP)/CUBE_PIXEL_LEN)**2  * devika_pulp_white_pixels) /
                                         ((math.cbrt(CUBE_TOOTH)/CUBE_PIXEL_LEN)**2 * devika_tooth_white_pixels))
    devika_pulp_to_tooth_scale_ratio  = (
        (mmpx_pulp**2  * devika_pulp_white_pixels) /
        (mmpx_tooth**2 * devika_tooth_white_pixels)
    )
    
    ptr = (
        devika_pulp_white_pixels /
        devika_tooth_white_pixels
    ) * (
        (SCALE_MM_PULP * devika_tooth_scale_px) /
        (SCALE_MM_TOOTH * devika_pulp_scale_px)
    ) ** 2

    age_pred = 54.32 - (554.21 * ptr)

    print("Devika tooth white pixels: " , devika_tooth_white_pixels)
    print("Devika pulp white pixels: "  , devika_pulp_white_pixels)
    print("\nDevika PTTR (pixel)\t:\t"  , devika_pulp_to_tooth_pixel_ratio)
    print("Devika PTTR (volume)\t:\t"   , devika_pulp_to_tooth_volume_ratio)
    print("Devika PTTR (scale)\t:\t"    , devika_pulp_to_tooth_scale_ratio)
    print("Devika PTTR (normal)\t:\t"   , ptr)
    print("Devika age  (true)\t:\t"     , AGE_TRUE_LABEL)
    print("Devika age  (pred)\t:\t"     , age_pred)
    
    # cv2.imshow("devika_pulp", devika_pulp)
    # cv2.imshow("devika_tooth", devika_tooth)
    
    # _ = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

def detect_pixels(image):
    _, w = image.shape
    roi  = image[:, int(w * 0.80):]

    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(
        edges              ,
        1                  ,
        np.pi / 180        ,
        threshold     = 20 ,
        minLineLength = 50 ,
        maxLineGap    = 5
    )

    if lines is None:
        return None

    candidates = []
    for x1, y1, x2, y2 in lines:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy > 5 * dx:
            candidates.append(dy)

    return max(candidates, default=None)


if __name__ == "__main__":
    main()

"""
FOOTNOTE OBSERVATION:

[ ] - Maunally removed the BG, and counted the no. of pixels.
[ ] - Put in our handy-dandy normalized formula for the rato.
[ ] - Devika's PTTR (normalized) -> 0.254
[ ] - Cannot use traditional formula for age, need a model.
[ ] - Have to try interpolation.

:D
"""
