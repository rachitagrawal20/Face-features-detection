#libraries required
import dlib
import cv2
import imutils
import numpy as np
from google.colab.patches import cv2_imshow

from google.colab import drive


#mouting drive to collab
#set root path according to your system
drive.mount('/content/gdrive')
root_path = 'utils/'


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it to the format (x, y, w, h)
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)
 
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def add_facial_features(rects, image, predictor):

  for (i, rect) in enumerate(rects):
    shap = predictor(gray, rect)
    shape = shape_to_np(shap)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
      cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return image

def add_jawline(rects, image):
  for (i, rect) in enumerate(rects):
    shap = predictor(gray, rect)
    shape = shape_to_np(shap)
    colors = [(180, 42, 220)]

    overlay = image.copy()
    alpha = 0.75
    pts = shape[0:17]
      # check if are supposed to draw the jawline
    for l in range(1, len(pts)):
      ptA = tuple(pts[l - 1])
      ptB = tuple(pts[l])
      cv2.line(overlay, ptA, ptB, colors[0], 2)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(root_path+"shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(root_path + "DP.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

#to add facial features
image = add_facial_features(rects, image, predictor)

#to add the jaw line
image = add_jawline(rects, image)

#saving the processed image
cv2.imwrite(root_path+"features.jpg", image)

cv2_imshow(image)
cv2.waitKey(0)