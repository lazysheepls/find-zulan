# Find Zulan in BangBang

Where is zulan? Can you spot him/her in the video? 



# Video

<video controls src="final.mp4" title="Title"></video>
[太强了！王祖蓝一人模仿结石姐&麻辣鸡&A妹演唱《Bang Bang》](https://www.bilibili.com/video/BV1HK4y1N718/?spm_id_from=333.337.search-card.all.click&vd_source=d531a4ab9358864c713e21e0248418f2) 但 YOLOV8


# Model

Based on Ultralytics YOLO V8.2 nano, trained on custom dataset of two classes(`zulan` and `not_zulan`). Dataset is prepared by roboflow.

# Takeaways

## Pre-processing is the key

In the dataset folder, you will find 8 datasets from `FindZulan-2` to `FindZulan-9`.

## Using image augmentation to boost dataset size

### What augmentation methods did we use?

Origin: dataset size increase from 30 images to 100+ images from `FindZulan2` to `FindZulan6`.


From `FindZulan7`: use **mosaic**, dataset size increased to 300+ images.


From `FindZulan8`: use **1% noise**, dataset size unchanged.

### Conclusion

Increase size of the dataset from 100+ to 300+ images largely improves the model performance. The mAP and the loss curve in result.png converged much better. Whereas adding the noise improves mAP50-95 only, given that mAP50 has reached maximum.


## Number of epochs

Model performance does not improve after 200 epochs.

## Define class (it matters)

### `Zulan`

Trained with web scrawled Zulan Wang images. Notably, adding Zulan images from 百變大咖秀 improves the model's generalization performance. 

Why? Because the face-swap zulan can be Jessica J Zulan, Nicki Minaj Zulan, Taylor Swift Zulan, New Zealand Lorde Zulan, Sam Smith Zulan. In all, all gender and all skin.


### `None-zulan`

This is tricky one. 

#### Bad attempt: with only 1 class
Not having `None-zulan` class is a no no. The model recognized any human face as `zulan`. 

#### Bad attempt: train `None-zulan` with general face collection

Using a general collection of human faces to trained the `None-zulan` class ended badly. As the model likely to identify face swapped `zulan` as `non-zulan`. It confuses the model.  

#### Balance is the key

In our video, most of the faces are swapped to the one from `zulan` expect for the members from the **Pentatonix** group. Thus, the `non-zulan` class is trained with **only** their photos. It worked much better than the previous attempt.

To conclude, selecting the right training dataset is the key. We don't want it to be too specific, nor too general.

