# Automatic Visual Traffic Monitoring

In order to run this solution, follow the steps below:
1. Install the dependencies by running

```sh

pip install -r requirements.txt

```

2. For task 1, open `task1.ipynb` and change `images_path` and `out_path` in the last block accordingly, like below:

```python

get_formatted_predictions('../train/Task1/', '../submission/Task1/')

```
Run all blocks afterwards.
3. For task2, open `task2.ipynb` and change `videos_path`, `out_path` and the YOLOv5s path in the last block accordingly, like below: 
```python

net = cv2.dnn.readNet('assets/yolov5s.onnx')

get_all_predictions('../train/Task2/', '../submission/Task2/', net)

```
Run all blocks afterwards.