import math
import cv2

cube_tooth = 29.33
cube_pulp  = 14.56

savitha_tooth = cv2.imread("kari/51-60/females/Savitha@52/tooth.bmp", cv2.IMREAD_GRAYSCALE)
savitha_pulp  = cv2.imread("kari/51-60/females/Savitha@52/pulp.bmp", cv2.IMREAD_GRAYSCALE)

savitha_tooth_white_pixels = cv2.countNonZero(savitha_tooth)
savitha_pulp_white_pixels  = cv2.countNonZero(savitha_pulp)

savitha_pulp_to_tooth_pixel_ratio = savitha_pulp_white_pixels / savitha_tooth_white_pixels
savitha_pulp_to_tooth_ratio = ((math.cbrt(cube_pulp)**2 * savitha_pulp_white_pixels) / (math.cbrt(cube_tooth)**2 * savitha_tooth_white_pixels))

print("Savitha tooth white pixels: " , savitha_tooth_white_pixels)
print("Savitha pulp white pixels: "  , savitha_pulp_white_pixels)
print("\nSavitha PTTR (pixel)\t:\t"  , savitha_pulp_to_tooth_pixel_ratio)
print("Savitha PTTR (normal)\t:\t"   , savitha_pulp_to_tooth_ratio)
print("Savitha age  (true)\t:\t 52")

cv2.imshow("savitha_pulp", savitha_pulp)
cv2.imshow("savitha_tooth", savitha_tooth)

_ = cv2.waitKey(0)
cv2.destroyAllWindows()

"""
FOOTNOTE OBSERVATION:

:D
"""
