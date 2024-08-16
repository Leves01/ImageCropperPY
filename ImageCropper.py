import cv2 as cv
import numpy as np
import os

# test_path = 'D:/SoulStoneWiki/Test'
path = 'D:\SoulStoneWiki\Achivements\Active Skill Achivements'

files = os.listdir(path)

for file in files:
    imagePath = f'{path}/{file}'

    img_rgb = cv.imread(imagePath)
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # Insert the wanted template.png into the array
    template_array = ['D:/SoulStoneWiki/square_template.png','D:/SoulStoneWiki/penta_template.png']

    # Change the number depending of which template in the array you want to use (0 is first)
    template = cv.imread(template_array[0], cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"

    edges_main = cv.Canny(img_gray, 20, 20)

    edges_template = cv.Canny(template, 20, 20)

    # Image testing
    # cv.imshow('Edges Main', edges_main)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imshow('Edges Main', edges_template)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # w, h = template.shape[::-1]
    w = 550
    h = 90

    res = cv.matchTemplate(edges_main, edges_template, cv.TM_CCOEFF_NORMED)

    # Thresholds for perfect cuts with these settings
    # Rectangle threshold 0.25-0.3
    # Pentagon threshold 0.4
    threshold = 0.3
    loc = np.where(res >= threshold)

    crop_count = 0
    detected_points = []

    for pt in zip(*loc[::-1]):

        if not any(abs(pt[0] - dp[0]) < w and abs(pt[1] - dp[1]) < h for dp in detected_points):
            detected_points.append(pt)

    detected_points = sorted(detected_points, key=lambda x: (x[1], x[0]))

    for i, pt in enumerate(detected_points[:10]):
        # Adjust crop region
        x_start = max(0, pt[0])
        y_start = max(0, pt[1])
        cropped_img = img_rgb[y_start:y_start + h, x_start:x_start + w]

        # Save cropped image
        crop_filename = f'D:/SoulStoneWiki/Results/crop_{i}_{files.index(file)}.png'
        cv.imwrite(crop_filename, cropped_img)
        print(f'Cropped image saves as: {crop_filename}')