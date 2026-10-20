# Stereo ChArUco Capture And Crop Design

## Purpose

Provide two independent programs. The first collects unmodified side-by-side
stereo ChArUco images. The second crops those images into paired left and right
monocular data sets for calibration.

## Design

`capture_stereo.py` opens one configurable OpenCV camera whose frame contains
the left and right views side by side. It splits frames only in memory to check
that both views contain enough ChArUco markers, then saves the original full
2560 x 720 frame without annotations as `stereo_NNN.jpg` in `stereo_images/`.
The camera uses Motion-JPEG at 30 fps to support this resolution over USB.

`crop_stereo.py` reads the saved stereo images and splits each one at its
horizontal midpoint, producing two 1280 x 720 images. It writes paired images
with identical file names to `charuco_left/` and `charuco_right/`. An explicit
split position and left/right swap are available for cameras with a non-default
layout.

`calib_charuco.py` derives its timestamped output directory from the input
directory name. For example, `charuco_left/` produces
`charuco_left_calib_<timestamp>/`, while the parameter and report file names
inside that directory remain unchanged.

The delete operation considers only files that match `stereo_NNN.jpg`. It
removes the file with the greatest numeric index, then reuses that index on the
next save. It never removes unrelated images. Existing matching images are
loaded on startup, which allows an interrupted capture session to continue.

## Controls And Errors

- `s`: save the current raw frame when marker detection succeeds.
- `d`: delete the latest matching saved image.
- `Esc` or `q`: release the camera and exit.

The program reports camera-open, camera-read, image-write, and image-delete
failures. It always releases the camera and closes the OpenCV window.

## Verification

Compile both modules, inspect their command-line help, test save/delete file
selection, and crop a synthetic image to verify dimensions and pixel ownership.
