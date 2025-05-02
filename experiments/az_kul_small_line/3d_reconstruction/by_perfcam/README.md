# Dataset Taken From PerfCam's Robotic Camera

This repository provides a dataset collected using the PerfCam's camera system mounted on a 3-axis robotic arm. The PerfCam's robotic camera integrates an Intel RealSense D435i camera to capture RGB and depth images, as well as IMU data. Additionally, each data point is accompanied by the corresponding motor angles of the PerfCam’s servo motors.

## Overview

The dataset consists of multiple folders, each representing a unique camera position or orientation of the PerfCam system. Within each position folder, you will find a series of files corresponding to individual frames captured at that position.

## Dataset Structure

Each position folder is named with a numeric identifier (e.g., `1`, `2`, `3`, ...) under `spots` folder. Inside each folder, there is a `data` folder in which the dataset contains four types of files per frame:

1. **RGB Image**: `frame_color_X.png`  
   - Standard RGB image captured from the Intel RealSense D435i.
   - `X` represents the frame index for that specific position.

2. **Depth Image**: `frame_depth_X.png`  
   - Depth information recorded using the camera’s depth sensor.
   - Same frame index `X` as the corresponding RGB image.

3. **IMU Data**: `frame_imu_X.json`  
   - JSON file containing accelerometer and gyroscope readings for that frame.
   - Provides information on camera motion and orientation during capture.  
   - Example format:
     ```
     {  
       "acceleration": {  
         "x": ...,  
         "y": ...,  
         "z": ...  
       },  
       "gyroscope": {  
         "x": ...,  
         "y": ...,  
         "z": ...  
       }  
     }
     ```

4. **Motor Angles**: `motor_angles_X.json`  
   - JSON file containing the angular positions (in degrees or radians) of the three servo motors controlling the PerfCam’s 3-axis arm.  
   - Example format:
   ```
     {  
       "angle0": ...,  
       "angle1": ...,  
       "angle2": ...  
     }
   ```

### Example Directory Layout

Here is the folder structure of folders inside `spots` folder:

```
1/data/
  frame_color_1.png
  frame_depth_1.png
  frame_imu_1.json
  motor_angles_1.json

  frame_color_2.png
  frame_depth_2.png
  frame_imu_2.json
  motor_angles_2.json
  ...

2/data/
  frame_color_1.png
  frame_depth_1.png
  frame_imu_1.json
  motor_angles_1.json
  ...
```

## Usage

- **Computer Vision Research**: The RGB and depth images can be used for tasks such as object recognition, 3D reconstruction, or scene understanding.
- **Robotics**: The IMU data combined with the known servo angles provides ground truth for robotic positioning, motion planning, and calibration.
- **Sensor Fusion**: The synchronized nature of RGB, depth, and IMU data supports sensor fusion experiments and SLAM (Simultaneous Localization and Mapping) tasks.

## Notes

- The dataset aims to provide consistent, time-synchronized data: each frame index `X` corresponds to a single capture instance of RGB, depth, IMU, and motor angles.
- Ensure that you interpret angles and IMU values correctly (e.g., units, coordinate frames) before using the data.
- The camera used (Intel RealSense D435i) provides intrinsic parameters that may be required for certain calibration and image processing tasks. Refer to the camera’s official documentation for exact specifications.

## License

Please refer to the `LICENSE` file in the root folder of the repository for licensing details.

## Contact

If you have any questions or require additional information about this dataset, please open an issue on GitHub or contact the repository maintainers.
