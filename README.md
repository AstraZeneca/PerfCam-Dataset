# Dataset for PerfCam: Digital Twinning for Production Lines Using 3D Gaussian Splatting and Vision Models

**[KTH Royal Institute of Technology, SCI](https://www.kth.se/en/sci/skolan-for-teknikvetenskap-1.795005)**; **[AstraZeneca, Sweden Operations](https://www.astrazeneca.com/)**

[Michel Gokan Khan](https://michelgokan.github.io/), [Renan Guarese](https://renghp.github.io/), [Fabian Johonsson](https://se.linkedin.com/in/fabianmartinjohnson), [Xi Vincent Wang](https://www.kth.se/profile/wangxi), [Anders Bergman](https://se.linkedin.com/in/anders-bergman-186203), [Benjamin Edvinsson](https://se.linkedin.com/in/benjamin-edvinsson-860ba968), [Mario Romero Vega](https://www.kth.se/profile/marior), [Jérémy Vachier](https://github.com/jvachier), [Jan Kronqvist](https://www.kth.se/profile/jankr)

[[`Paper`](#)] [[`Project`](https://www.digitalfutures.kth.se/research/industrial-postdoc-projects/smart-smart-predictive-maintenance-for-the-pharmaceutical-industry/)] [[`BibTeX`](#citing-perfcam-dataset)]


![402361172-5fa3ae9f-5d48-43c2-9d5c-9e2a3c4eb807](https://github.com/user-attachments/assets/1b79665a-2188-43f0-9af4-36d14c5caf4f)


**PerfCam** is an open-source proof of concept that integrates 3D Gaussian Splatting with real-time object detection to achieve precise digital twinning of industrial production lines. This approach leverages existing camera systems for both 3D reconstruction and object tracking, reducing the need for additional sensors and minimizing initial setup and calibration efforts. 

This repository presents the dataset used in the PerfCam's original paper. This dataset is to support further research in the area of insdustrial 3D reconstruction, digital twinning, and predictive maintenance.

## Dataset

https://github.com/user-attachments/assets/7120515f-0ab8-462b-b11f-c3b36073de83

Here you can read more about different topics in this dataset:

### 3D Reconstruction
[Dataset Generated From PerfCam's Robotic Camera](experiments/az_kul_small_line/3d_reconstruction/by_perfcam/)

[Imaegs Taken Using A Pixel 7 Pro](experiments/az_kul_small_line/3d_reconstruction/by_phone)

### Experiment at AZ Kul

[Details of the Experiment and Events](experiments/az_kul_small_line/object_and_event_detection)

[Dataset for Trained YOLO Model](experiments/az_kul_small_line/object_and_event_detection/trained)

## COLMAP Point Clouds

### Raw Output (after feature extraction)

https://github.com/user-attachments/assets/4e1a119a-2352-481a-9b75-cd62e102288b

### Fused Output

https://github.com/user-attachments/assets/7bcd3374-6998-4cfb-abfb-24b4bd051144

## LFS Considerations
To optimize this repository's performance and ensure efficient handling of large files, make sure you track the following file types with Git LFS:

```
git lfs track **/*.dng
git lfs track **/*.mp4
git lfs track **/*.jpg
git lfs track **/*.bin 
git lfs track **/*.ply
git lfs track **/*.pt 
git lfs track **/*.mov
git lfs track **/*.obj
git lfs track **/*.bytes
git lfs track **/*.png
git lfs track **/*.tar.gz*
git lfs track **/*.mtl
git lfs track **/*.log
git lfs track **/*.log.csv
git lfs track **/*.csv
```

## Repostiory Contributors
- [Michel Gokan Khan](https://github.com/michelgokan) (Main Contributor)

## License
- All files in this repository are licensed under the [Apache-2.0 License](LICENSE) **except** the YOLO weights under `experiments/*/object_and_event_detection/trained/model/train/weights`, which are licensed under the AGPL (look for a LICENSE-YOLO-AGPL inside the folder next to the weight files, see [this example](experiments/az_kul_small_line/object_and_event_detection/trained/model/train/weights/LICENSE-YOLO-AGPL)). 


## Citing PerfCam Dataset
If you use PerfCam or PerfCam Dataset in your research, please use the following BibTeX entry.
```
@article{perfcam,
  title={PerfCam: Digital Twinning for Production Lines Using 3D Gaussian Splatting and Vision Models},
  author={Michel Gokan Khan and Renan Guarese and Fabian Johonsson and Xi Vincent Wang and Anders Bergman and Benjamin Edvinsson and Mario Romero Vega and Jeremy Vachier and Jan Kronqvist},
  journal={TBA},
  year={2025}
}

@data{73cd-3668-25,
  doi = {10.21227/73cd-3668},
  url = {https://dx.doi.org/10.21227/73cd-3668},
  author = {Gokan Khan, Michel and Guarese, Renan and Johnson, Fabian and Wang, Xi and Romero, Mario and Vachier, Jérémy and Kronqvist, Jan},
publisher = {IEEE Dataport},
  title = {Experiments Dataset for PerfCam: Digital Twinning for Production Lines Using 3D Gaussian Splatting and Vision Models},
  year = {2025} 
}
```
