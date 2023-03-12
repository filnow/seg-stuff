import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

MASK_COLOR = (0, 255, 0) # green mask

cap = cv2.VideoCapture(0)

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:

  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    fg_image = cv2.bitwise_and(image, fg_image)

    output_image = np.where(condition, fg_image, image)

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()