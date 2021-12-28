#!/usr/bin/python3

# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# MQTT
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

# For measuring the inference time.
import time

# Scheduller
from apscheduler.schedulers.blocking import BlockingScheduler

# cv
import cv2 as cv

# client = mqtt.Client(client_id="TensorFlow Motion Detection")

# Config constants
MQTT_SERVER = '10.238.75.62'
IMG_TMP = '/tmp/hatfimg.jpg'

def getPic():
  cap = cv.VideoCapture('rtsp://server:server@10.238.75.200:554') # it can be rtsp or http stream

  ret, frame = cap.read()

  if cap.isOpened():
      _,frame = cap.read()
      cap.release() #releasing camera immediately after capturing picture
      if _ and frame is not None:
          cv.imwrite(IMG_TMP, frame)

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  # plt.imshow(image)
  # plt.imsave("./boxed.jpg", image)
  # plt.show()


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image

image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"

# downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)
downloaded_image_path = IMG_TMP
# module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"


# detector = hub.load(module_handle)
detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  # converted_img = converted_img[:, :, :, :3]
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

#   image_with_boxes = draw_boxes(
#       img.numpy(), result["detection_boxes"],
#       result["detection_class_entities"], result["detection_scores"])
  presence = "false"
  for i in range(0, len(result["detection_scores"])):
      className = result["detection_class_entities"][i].decode("ascii")
      if "Person" in className:
          print("Presence + score", result["detection_scores"][i])
          if result["detection_scores"][i] > 0.1:
            presence = "true"
            break
  publish.single("hatf/motion_living_room/state", payload=presence, qos=2, hostname=MQTT_SERVER, client_id="TensorFlow Presence Detection", auth={'username': 'mqtt', 'password': 'mqtt'})


#   display_image(image_with_boxes)
#   image_with_boxes_img = Image.fromarray(image_with_boxes)
#   image_with_boxes_rgb = image_with_boxes_img.convert("RGB")
#   image_with_boxes_rgb.save("./boxed.jpg", format="JPEG", quality=90)

# while True:
#   getPic()
#   run_detector(detector, downloaded_image_path)
#   readchar.readchar()

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.

def on_message(client, userdata, msg):
    pass

def detect():
    print("Detect")
    getPic()
    run_detector(detector, downloaded_image_path)

if __name__ == '__main__':
    global client
    # Scheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(detect, 'interval', seconds=10)

    # MQTT
    client = mqtt.Client(client_id="TensorFlow Motion Detection")
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set("mqtt", "mqtt")
    client.connect(MQTT_SERVER, 1883, 60)
    client.loop_start()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        publish.single("hatf/motion_living_room/state", payload="true", qos=2, hostname=MQTT_SERVER, client_id="TensorFlow Presence Detection", auth={'username': 'mqtt', 'password': 'mqtt'})
        client.disconnect()
        pass
