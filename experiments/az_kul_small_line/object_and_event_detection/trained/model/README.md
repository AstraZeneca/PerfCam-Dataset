# Training the PerfCam's Experiment 1's Dataset with Yolov11

## Description

This model has been trained using Yolov11 based on the following parameters:

```
MODEL_NAME = "yolov11"
ULTRALYTICS_MODEL_NAME = "yolo11x"
LABELS = ['product']
EPOCS_COUNT = 200
MODEL = "train/weights/best.pt"
```

However, the training stopped at epoch 133 as no improvement observed in last 100 epochs.

> [!NOTE]
> For reusability and improved accuracy, this model is trained on top of a pre-trained Yolo model trained by Ultralytics that has been trained on [COCO](https://cocodataset.org/), which include 80 pre-trained classes. [[Reference]](https://github.com/ultralytics/ultralytics)
> | Model     | Size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
> |-----------|---------------|--------------|----------------------|--------------------------|------------|-----------|
> | YOLO11x   | 640           | 54.7         | 462.8 ± 6.7          | 11.3 ± 0.2               | 56.9       | 194.9     |


The model has been trained on a Nvidia A100-SXM3-40GB.

```
dict_keys(['date', 'version', 'license', 'docs', 'epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'train_args', 'train_metrics', 'train_results'])
Ultralytics 8.3.55 🚀 Python-3.10.12 torch-2.5.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB, 40514MiB)
engine/trainer: task=detect, mode=train, model=/content/yolo11x.pt, data=/content/drug-counting-3/data.yaml, epochs=200, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...
100% 755k/755k [00:00<00:00, 14.5MB/s]
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1      2784  ultralytics.nn.modules.conv.Conv             [3, 96, 3, 2]                 
  1                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  2                  -1  2    389760  ultralytics.nn.modules.block.C3k2            [192, 384, 2, True, 0.25]     
  3                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
  4                  -1  2   1553664  ultralytics.nn.modules.block.C3k2            [384, 768, 2, True, 0.25]     
  5                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  6                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  7                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
  8                  -1  2   5022720  ultralytics.nn.modules.block.C3k2            [768, 768, 2, True]           
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]                 
 10                  -1  2   3264768  ultralytics.nn.modules.block.C2PSA           [768, 768, 2]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  2   1700352  ultralytics.nn.modules.block.C3k2            [1536, 384, 2, True]          
 17                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  2   5317632  ultralytics.nn.modules.block.C3k2            [1152, 768, 2, True]          
 20                  -1  1   5309952  ultralytics.nn.modules.conv.Conv             [768, 768, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  2   5612544  ultralytics.nn.modules.block.C3k2            [1536, 768, 2, True]          
 23        [16, 19, 22]  1   3146707  ultralytics.nn.modules.head.Detect           [1, [384, 768, 768]]          
YOLO11x summary: 631 layers, 56,874,931 parameters, 56,874,915 gradients, 195.4 GFLOPs

Transferred 1009/1015 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir runs/detect/train', view at http://localhost:6006/
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt to 'yolo11n.pt'...
100% 5.35M/5.35M [00:00<00:00, 65.6MB/s]
AMP: checks passed ✅
train: Scanning /content/PerfCam-3/train/labels... 440 images, 40 backgrounds, 0 corrupt: 100% 440/440 [00:00<00:00, 1097.66it/s]
train: New cache created: /content/PerfCam-3/train/labels.cache
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 658, len(boxes) = 1908. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
val: Scanning /content/PerfCam-3/valid/labels... 32 images, 1 backgrounds, 0 corrupt: 100% 32/32 [00:00<00:00, 1039.71it/s]
val: New cache created: /content/PerfCam-3/valid/labels.cache
Plotting labels to runs/detect/train/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.002, momentum=0.9) with parameter groups 167 weight(decay=0.0), 174 weight(decay=0.0005), 173 bias(decay=0.0)
TensorBoard: model graph visualization added ✅
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/detect/train
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/200      16.5G      2.137      3.257      1.909         52        640: 100% 28/28 [00:09<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 1/1 [00:01<00:00,  1.52s/it]
                   all         32        147          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/200      16.7G      1.991      2.024      1.777         37        640: 100% 28/28 [00:07<00:00,  3.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  5.61it/s]
                   all         32        147          0          0          0          0

      .
      .
      .
 
     
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    132/200      16.4G     0.9219     0.4976       1.09         42        640: 100% 28/28 [00:06<00:00,  4.07it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  4.20it/s]
                   all         32        147      0.847      0.716      0.784       0.46

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    133/200      16.4G     0.9147     0.4882      1.094         66        640: 100% 28/28 [00:06<00:00,  4.07it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  4.47it/s]
                   all         32        147      0.866       0.68      0.761      0.449


EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 33, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

133 epochs completed in 0.335 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 114.4MB
Optimizer stripped from runs/detect/train/weights/best.pt, 114.4MB

Validating runs/detect/train/weights/best.pt...
Ultralytics 8.3.55 🚀 Python-3.10.12 torch-2.5.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB, 40514MiB)
YOLO11x summary (fused): 464 layers, 56,828,179 parameters, 0 gradients, 194.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 1/1 [00:00<00:00,  3.60it/s]
                   all         32        147       0.94       0.68      0.806      0.508
Speed: 0.1ms preprocess, 4.4ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to runs/detect/train
```

## Training Results

The following visualizations and metrics were generated to evaluate the performance of the YOLOv11 model during training:

1. **Precision-Confidence Curve**
   - **Purpose**: To illustrate how precision changes with varying confidence threshold levels.
   - **Explanation**: Precision is the ratio of true positive predictions to the total predicted positives. This curve helps identify the optimal confidence threshold that maximizes precision, indicating the reliability of positive predictions by the model.
   - ![Precision-Confidence Curve](val/P_curve.png)

2. **Precision-Recall Curve**
   - **Purpose**: To show the trade-off between precision and recall at different threshold settings.
   - **Explanation**: This curve helps assess the model's ability to balance between not missing any relevant instances (high recall) and not including too many false positives (high precision). A higher area under the curve (AUC) is indicative of better performance.
   - ![Precision-Recall Curve](val/PR_curve.png)

3. **Recall-Confidence Curve**
   - **Purpose**: To depict how recall varies with changes in confidence thresholds.
   - **Explanation**: Recall, also known as sensitivity, is the ratio of true positive predictions to the total actual positives. This curve helps determine the confidence threshold that captures the most true positives without excessively increasing false negatives.
   - ![Recall-Confidence Curve](val/R_curve.png)

4. **F1-Confidence Curve**
   - **Purpose**: To display the F1 score across different confidence thresholds.
   - **Explanation**: The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both. This curve assists in identifying the confidence level that optimizes the balance between precision and recall.
   - ![F1-Confidence Curve](val/F1_curve.png)

5. **Confusion Matrix Normalized**
   - **Purpose**: To provide a normalized view of the confusion matrix.
   - **Explanation**: The confusion matrix shows the number of true positive, true negative, false positive, and false negative predictions. Normalization adjusts these counts to proportions, making it easier to compare performance across classes.
   - ![Confusion Matrix Normalized](val/confusion_matrix_normalized.png)

6. **Labels Correlogram**
   - **Purpose**: To visualize the spatial correlation between labeled features.
   - **Explanation**: A correlogram is typically used to depict the relationship and correlation between different variables. In this context, it likely shows how the coordinates (x, y) and dimensions (width, height) of labeled objects correlate, which can be useful in understanding spatial distribution patterns in the dataset.
   - ![Labels Correlogram](train/labels_correlogram.jpg)


## Validation Batches

 1. **Validation Batch 0**
   - **Ground Truth Labels**: This image shows the true labels for objects in the validation batch 0.
     - ![Validation Batch 0 Labels](val/val_batch0_labels.jpg)
   - **Model Predictions**: This image showcases the predictions made by the model on validation batch 0, allowing for a visual comparison with the ground truth.
     - ![Validation Batch 0 Predictions](val/val_batch0_pred.jpg)

2. **Validation Batch 1**
   - **Ground Truth Labels**: This image displays the true labels for objects in the validation batch 1.
     - ![Validation Batch 1 Labels](val/val_batch1_labels.jpg)
   - **Model Predictions**: This image illustrates the predictions made by the model on validation batch 1, facilitating a visual comparison with the actual labels.
     - ![Validation Batch 1 Predictions](val/val_batch1_pred.jpg)


