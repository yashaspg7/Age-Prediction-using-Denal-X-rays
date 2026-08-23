import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUBE_VOLUME = 200


def main():
    df = pd.read_csv("./data_cbct/normalisation.csv")
    
    results           = []
    real_area_results = []
    
    for record in df.itertuples():
        cube_volume_pulp  = record.cube_volume_pulp
        cube_volume_tooth = record.cube_volume_tooth
        
        scale_length_pulp  = record.scale_length_pulp
        scale_length_tooth = record.scale_length_tooth

        img_pulp           = cv2.imread(record.image_location + "/pulp.bmp",           cv2.IMREAD_GRAYSCALE)
        img_tooth          = cv2.imread(record.image_location + "/tooth.bmp",          cv2.IMREAD_GRAYSCALE)
        img_pulp_original  = cv2.imread(record.image_location + "/pulp_original.bmp",  cv2.IMREAD_GRAYSCALE)
        img_tooth_original = cv2.imread(record.image_location + "/tooth_original.bmp", cv2.IMREAD_GRAYSCALE)
        
        scale_px_pulp  = (detect_pixels(img_pulp_original))
        scale_px_tooth = (detect_pixels(img_tooth_original))

        mmpx_pulp  = scale_length_pulp  / scale_px_pulp
        mmpx_tooth = scale_length_tooth / scale_px_tooth

        white_pixels_pulp  = cv2.countNonZero(img_pulp)
        white_pixels_tooth = cv2.countNonZero(img_tooth)
        
        ptr_pixel  = white_pixels_pulp / white_pixels_tooth
        ptr_cube_length = (
            ((((cube_volume_pulp) /CUBE_VOLUME)**(2)) * white_pixels_pulp) /
            ((((cube_volume_tooth)/CUBE_VOLUME)**(2)) * white_pixels_tooth)
        )
        ptr_scale_length = (
            ((mmpx_pulp **2) * white_pixels_pulp) /
            ((mmpx_tooth**2) * white_pixels_tooth)
        )
        
        results.append({
            "age"              : record.age,
            "sex"              : record.sex,
            "ptr_naive"        : ptr_pixel,
            "ptr_scale_length" : ptr_scale_length,
        })

        real_area_results.append({
            "age"        : record.age,
            "pulp_area"  : (mmpx_pulp **2) * white_pixels_pulp,
            "tooth_area" : (mmpx_tooth**2) * white_pixels_tooth
        })

        print("="*44)
        print("PULP:")
        print(f"\tSCALE_LENGTH :\t{scale_length_pulp}")
        print(f"\tPIXEL_SCALE  :\t{scale_px_pulp}")
        print(f"\tPIXEL_AREA   :\t{white_pixels_pulp}")
        print(f"\tREAL_AREA    :\t{(mmpx_pulp**2) * white_pixels_pulp}")
        print("TOOTH:")
        print(f"\tSCALE_LENGTH :\t{scale_length_tooth}")
        print(f"\tPIXEL_SCALE  :\t{scale_px_tooth}")
        print(f"\tPIXEL_AREA   :\t{white_pixels_tooth}")
        print(f"\tREAL_AREA    :\t{(mmpx_tooth**2) * white_pixels_tooth}")
        print("PTR:")
        print(f"\tnaive        :\t{ptr_pixel}")
        print(f"\tvol_normal   :\t{ptr_cube_length}")
        print(f"\tscale_normal :\t{ptr_scale_length}")
        
    results           = pd.DataFrame(results)
    real_area_results = pd.DataFrame(real_area_results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(real_area_results.age, real_area_results.pulp_area, color="royalblue", label="PULP")
    ax.scatter(real_area_results.age, real_area_results.tooth_area, color="darkorange", label="TOOTH")
    ax.set_title("pulp/tooth Real Area")

    ax.set_xlabel("AGE")
    ax.set_ylabel("REAL_AREA (PULP/TOOTH)")
    ax.grid(alpha=0.25)
    ax.legend()

    colors = {"female": "hotpink", "male": "dodgerblue"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for sex, color in colors.items():
        data = results[results.sex == sex]
        ax[0].scatter(data.age, data.ptr_naive, color=color, label=sex)
        ax[1].scatter(data.age, data.ptr_scale_length, color=color, label=sex)

    ax[0].set_title("Raw Pixel PTR")
    ax[1].set_title("Scale-normalized PTR")

    for a in ax:
        a.set_xlabel("Age")
        a.set_ylabel("Pulp / Tooth Ratio")
        a.grid(alpha=0.25)
        a.legend()

    plt.tight_layout()
    plt.show()


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
