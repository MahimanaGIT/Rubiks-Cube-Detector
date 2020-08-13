# Rubiks-Cube-Detection Using Classical Computer Vision Techniques

Implementation: https://youtu.be/tbmGO0xq0gY

This is the tutorial to detect a Rubiks Cube using computer vision. OpenCV is used as a framework. 

3*3*3 and 4*4*4 Rubik's Cube were used for implementation. We used classical computer vision techniques such as thresholding, contour detection, shape approximation, hough transform, optical flow for the detection of the cube.

The steps of implementation are as follow:
1. Convert the image in gray scale
2. Detect edges from the image using canny edge detector
3. Find closed contours in the image
4. Approximate contour shape and apply threshold on the basis of detection quadriletrals
5. Compute overall centroid of the contours
6. Using nearest neighbour algorithm, extrapolate the detection of contours to predict undetected squares
7. Detect color of Squares
8. Apply optical flow to detect motion of the cube
