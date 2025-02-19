The dataset includes 512 images.
PerfCam are annotated in YOLOv11 format via RoboFlow.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Grayscale (CRT phosphor)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 10 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random Gaussian blur of between 0 and 1.3 pixels
* Salt and pepper noise was applied to 1.13 percent of pixels


