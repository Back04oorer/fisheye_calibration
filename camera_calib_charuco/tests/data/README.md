`charuco_stalled_corner.png` is a 61 × 61 grayscale crop of the right camera's
decoded pixels from diagnostic retake frame 483, ChArUco corner 35 (11 October
2026 local time). It contains only the printed pattern. The crop origin in the
right image is (1350, 663).

The selected CLAHE estimate was (33.151855, 33.455688) in this crop. A native
3-pixel-half-size `cornerSubPix` call returns that estimate unchanged: the true
correction exceeds its window. Native-image detection and refinement from nearby
seeds locate the corner at approximately (30.5546, 30.3564). This fixture catches
the silent failed refinement without needing camera hardware or a fitted model.

`charuco_skewed_corner.png` is a 135 × 144 grayscale crop of the left camera's
decoded pixels from diagnostic retake frame 262, ChArUco corner 68 (11 October
2026 local time). It contains only the printed pattern. The accompanying JSON
records the crop origin (1299, 677), native-image corner estimates and their IDs.
The board has 11 × 8 squares, 20 mm squares, 15 mm markers, DICT_5X5_100 and the
legacy layout.

The tilted target corner stays near (78.6482, 99.6437) with the native 3-pixel
search and its stalled-seed retry. A larger native window with nearby seeds
locates the printed junction near (79.5, 103); locally board-aligned refinement
locates it near (78.985, 102.767). The test uses the image and neighbouring
observations, without a fitted camera model or stereo error. A separate synthetic
test ensures neighbours orient the search without pulling a measured corner to
the ideal grid.

`charuco_marker_recovery.png` is a 476 × 448 grayscale crop from the left eye of
retake frame 344, at origin (1020, 453). It shows the printed board and a small
surrounding area. Native detection finds 38 markers and 52 corners. OpenCV's
board-based rejected-marker recovery finds 43 markers and 68 corners, preserving
all 52 original corner IDs. It exercises marker recovery in the tilted, blurred
left view without fitted camera parameters, temporal history, or hardware.
