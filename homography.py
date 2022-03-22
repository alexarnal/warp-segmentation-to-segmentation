#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:36:42 2022
Copied from https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/features2D/Homography/perspective_correction.py

@author: latente
"""

from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

def getContourCoords(img):
    #x,y=[],[]
    _, threshold = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
    contours, _= cv.findContours(threshold[:,:,0], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    idx = np.round(np.linspace(0, len(contours[0]) - 1, 50)).astype(int)
    c = contours[0][idx]
    c = c[np.lexsort(c)][0]
    return True, c
        
def randomColor():
    color = np.random.randint(0, 255,(1, 3))
    return color[0].tolist()

def perspectiveCorrection(img1Path, img2Path):
    img2 = cv.imread(img2Path)
    img1 = cv.resize(cv.imread(img1Path), img2.shape[:2][::-1]) #resize experimental image to atlas dims
    
    # [find-corners]
    ret1, corners1 = getContourCoords(img1) 
    ret2, corners2 = getContourCoords(img2) 
    # [find-corners]
    
    if not ret1 or not ret2:
        print("Error, cannot find the chessboard corners in both images.")
        sys.exit(-1)

    # [estimate-homography]
    H, _ = cv.findHomography(corners1, corners2)
    # [estimate-homography]

    # [warp-chessboard]
    img1_warp = cv.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    # [warp-chessboard]

    img_draw_warp = cv.hconcat([img2, img1_warp])
    cv.imshow("Desired chessboard view / Warped source chessboard view", img_draw_warp )

    corners1 = corners1.tolist()
    corners1 = [a[0] for a in corners1]

    # [compute-transformed-corners]
    img_draw_matches = cv.hconcat([img1, img2])
    for i in range(len(corners1)):
        pt1 = np.array([corners1[i][0], corners1[i][1], 1])
        pt1 = pt1.reshape(3, 1)
        pt2 = np.dot(H, pt1)
        pt2 = pt2/pt2[2]
        end = (int(img1.shape[1] + pt2[0]), int(pt2[1]))
        cv.line(img_draw_matches, tuple([int(j) for j in corners1[i]]), end, randomColor(), 2)

    cv.imshow("Draw matches", img_draw_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
    # [compute-transformed-corners]

def main():
    img1Path = '../../brainRegions/20-005(slide3section7)(58)(VV)/processed/MARN.png' #'left02.jpg' #args.image1
    img2Path = '../../brainRegions/BM4-Complete-Atlas-Level-58/processed/MARN.png' #'left01.jpg' #args.image2
    perspectiveCorrection(img1Path, img2Path)

if __name__ == "__main__":
    main()