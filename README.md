## What is HabitBreaker?

HabitBreaker uses your webcam and a convolutional neural network to detect and stop you from preforming your bad habits.

[Demo Gif](https://media.giphy.com/media/7ThuRA3bPmBfrRynXn/giphy.gif)


## Table of Contents

- How was it made
- Dependencies
- Steps to Recreate

## How was it made
HabitBreaker is made using transfer learning. Transfer learning utilizes a fully trained model and retrains its existing weights to classify a new set of classes. In this case we will take the popular Inception-v3 model and retraining the final layer from scratch to classify our bad habit.

## Dependencies
- Tensorflow
- opencv-python
- win10toast
    - for windows notifications

## Steps to Recreate:

1. Capture normal activity. Remember variety greatly out weights quantity!
```
python captureData.py --split train --classification negative --fps 10
```

2. Capture you doing your bad habit.
```
python captureData.py --split train --classification nailbiting --fps 10
```

3. Train the model.
```
python tensorflow/examples/image_retraining/retrain.py \
    --bottleneck_dir=data/train/bottlenecks  \
    --output_graph=data/retrained_graph.pb  \
    --output_labels=data/retrained_labels.txt \
    --image_dir data/train \
    --how_many_training_steps 300
```

4. Capture one image just to check the model.
```
python captureData.py --one_img true
```

5. Make a prediction on the single image.
```
python singlePred.py --img data/single_imgs/img.jpg
```

6. Now run the model in real time.
```
python habitBreaker.py --fps -1 --notify nailbiting
```

7. Enable windows notifications by editing habbitBreaker.py
```python
    # Set to True if you want to receive notification via windows notification center
    windows_notify = True
```

Congratulations you are one step closer to breaking your bad habit!

**For more information visit my blog post:** [How I broke a bad habit using deep learning](https://www.coreyalexandersmith.com/projects/habit-breaker/)
