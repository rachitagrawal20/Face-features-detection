# Face-features Detection
### Prerequisites

```
dlib
cv2
numpy
imutils
```

### Notes

This project is about is getting the facial-landmarks from any image with multiple faces. You can also draw the curve around jawline using appropriate functions.

The functions declared include
```
add_facial_features(rects, image, predictor)
```
rects : faces detected using dlib detector \
image : the image you want to work on \
predictor : dlib predictor which can predict facial features given a detected face. Here I have used dlib’s pre-trained facial landmark detector [link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Download this pretrained model and move it into the `Utils` folder

```
add_jawline(rects, image)
```
rects :  faces detected using dlib detector \
image : the image you want to work on


### Caution
This project was implemented on Google Colab. Hence you will find some drive mounting code and google.colab libraries. Remove it, if you are running it locally.

### Contribution
This code was written with the help of [this](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) article. I thank the author for this clear and concise article.