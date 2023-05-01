import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def get_keypoints_and_features(image) -> tuple:
    """
    :param image.
    :return the keypoints: [cv.Keypoint] and the features: np.ndarray for each keypoint.
    """
    
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    
    keypoints, features = sift.detectAndCompute(gray_image, None)
        
    return keypoints, features

def align_images(img1, img2, good_match_percent=0.75, ref_filename=''):
    k1, f1 = get_keypoints_and_features(img1)
    k2, f2 = get_keypoints_and_features(img2)

    matcher = cv.DescriptorMatcher_create('FlannBased')
    # matches = matcher.match(f1, f2, None)
    all_matches = matcher.knnMatch(f1, f2, k=2) 

    # Add the best matches in a list
    matches = [match[0] for match in all_matches if match[0].distance / match[1].distance < good_match_percent]

    # list(matches).sort(key=lambda x: x.distance, reverse=False)
 
    # num_good_matches = int(len(matches) * good_match_percent)
    # matches = matches[:num_good_matches]

    # Draw top matches
    im_matches = cv.drawMatches(img1, k1, img2, k2, matches, None)
    if ref_filename != '':
        cv.imwrite("matches_" + ref_filename + ".jpg", im_matches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = k1[match.queryIdx].pt
        points2[i, :] = k2[match.trainIdx].pt

    # Find homography
    h, _ = cv.findHomography(points1, points2, cv.RANSAC)

    # Use homography
    height, width, _ = img2.shape
    im1_reg = cv.warpPerspective(img1, h, (width, height))

    return im1_reg, h

def save_aligned_image(ref_filename, image_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path) 

    im_reference = cv.imread(ref_filename, cv.IMREAD_COLOR)

    # Read image to be aligned
    im = cv.imread(image_path, cv.IMREAD_COLOR)

    # Registered image will be stored in im_reg.
    im_reg, _ = align_images(im, im_reference)

    # Crop image
    cropped_image = im_reg[273 * 2 : (273 + 960) * 2,
                           547 * 2 : (547 + 956) * 2]
    
    # Align cropped image to grid
    w, h = cropped_image.shape[0:2]
    pts1 = np.float32([[0,0],[1863,0],[5,1918],[1906,1880]]) 
    pts2 = np.float32([[0,0],[1911,0],[0,1919],[1911,1919]])
    M = cv.getPerspectiveTransform(pts1, pts2)

    warped_cropped_image = cv.warpPerspective(cropped_image, M, (w, h))
    # Write aligned image to disk.
    out_filename = "aligned_" + image_path.split('\\')[-1]
    cv.imwrite(out_path + out_filename, warped_cropped_image)

def save_aligned_images(ref_filename, images_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path) 

    print("Reading reference image : " + ref_filename)
    ref_img = cv.imread(ref_filename, cv.IMREAD_COLOR)
    cropped_ref_img = ref_img[273 * 2 : (273 + 960) * 2,
                              547 * 2 : (547 + 956) * 2]

    cv.imwrite(out_path + 'template.jpg', cropped_ref_img)

    # Read images to be aligned
    img_filenames = [images_path + f for f in os.listdir(images_path) if f.endswith('.jpg')]
    for idx, img_filename in enumerate(img_filenames):
        save_aligned_image(ref_filename, img_filename, out_path)
        progress_bar(idx, len(img_filenames))

def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-filepath', '-r', required=True, help='The path of the image to be taken as reference for the alignment')
    parser.add_argument('--images-path', '-p', required=True, help='The path where the files to be aligned are located')
    parser.add_argument('--out-path', '-o', required=True, help='The output path')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()

    ref_filepath = args.ref_filepath
    images_path = args.images_path
    out_path = args.out_path 

    save_aligned_images(ref_filepath, images_path, out_path)