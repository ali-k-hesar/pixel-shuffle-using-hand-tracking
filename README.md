<div align="center">
  <p>
    <img width="100%" src="https://github.com/ali-k-hesar/pixel-shuffle-using-hand-tracking/assets/85279433/273d59f3-2fa8-47c1-a951-2af9251345e8"></a>
  </p>
</div>

# Pixel Shuffling Using Hand Tracking

The "Pixel Shuffling Control Using Hand Tracking" project combines computer vision techniques with hand tracking to create an interactive image processing experience. The goal of this project is to allow users to control the degree of randomness in pixel shuffling effects using their hand position. By leveraging the power of YOLOv8m-segmentation for background removal and Mediapipe for hand tracking, the application provides real-time feedback and visual effects on a webcam feed.

The image processing pipeline consists of multiple stages. Firstly, background removal is performed using YOLOv8-segmentation to separate the foreground from the background. This ensures that the pixel shuffling effects only affect the foreground objects, creating a visually appealing and immersive experience. Next, hand tracking using Mediapipe allows the application to detect and track the user's hand movements in real-time. The position of the hand is then used to control the degree of randomness in the pixel shuffling effects. Additionally, the application includes image pixelation functionality, where the code provided by ChatGPT has been modified to achieve the desired pixelation effect. Lastly, the pixel shuffling code created by ChatGPT is incorporated, providing a captivating visual transformation of the image. By combining these techniques, users can explore different hand gestures and positions to dynamically control the randomness and visual outcome of the pixel shuffling effect.

## Installation

#### 1. Clone the repository:

```shell
git clone https://github.com/ali-k-hesar/pixel-shuffle-using-hand-tracking.git
```

#### 2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

#### 3. Download the background removal model weights and place them in the bg_models directory.

```shell
python main_python.py --model-type m --webcam-number 0
```
- **model-type** Choose between "n" for nano, "s" for small, "m" for medium, "l" for large and "x" for x-large
- **webcam-number** based on the total webcams connected to your computer (if only one webcam is connected choose 0)
- Press "q" to quit

<details><summary>Segmentation</summary>
  
See [YOLOv8](https://github.com/ultralytics/ultralytics) GitHub page for more info.
See [Segmentation Docs](https://docs.ultralytics.com/tasks/segmentation/) for usage examples with these models.

| Model                                                                                    | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.
  <br>Reproduce by `yolo val segment data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>Reproduce by `yolo val segment data=coco128-seg.yaml batch=1 device=0/cpu`

</details>

## Usage

1. Connect a webcam to your computer.
2. Run the application by executing the main_python.py script.
3. The webcam feed will open in a window, and the image processing effects will be applied in real-time.
4. Press 'q' to exit the application.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
