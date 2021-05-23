import cv2
import random
import numpy as np

MAX_ANGLE = 180
MAX_ELLIPSE = 10
MAX_RECTANGLE_SIZE = 15
MAX_COLOR = 255
MIN_COLOR = 0
NUM_ITERATIONS = 15000
NUM_INIT = 10000
NUM_IMG = 100

trg_path = "/Users/ryoya/Desktop/GA/data/akua.jpeg"
trg_img = cv2.imread(trg_path)
H, W, C = trg_img.shape
cv2.imwrite('target.jpg', trg_img)


def random_color():
    r = random.randint(MIN_COLOR, MAX_COLOR)
    g = random.randint(MIN_COLOR, MAX_COLOR)
    b = random.randint(MIN_COLOR, MAX_COLOR)
    return r, g, b


def random_coord():
    x = random.randint(0, W)
    y = random.randint(0, H)
    return x, y


def random_rectangle_plot(img):
    coord1 = random_coord()
    coord2 = list(coord1).copy()
    coord2[0] = random.randint(0, MAX_RECTANGLE_SIZE) + coord1[0]
    coord2[1] = random.randint(0, MAX_RECTANGLE_SIZE) + coord1[1]
    color = random_color()
    img = cv2.rectangle(img, coord1, coord2, color, -1)
    return img


def random_ellipse_plot(img):
    coord1 = random_coord()
    coord2 = list(coord1).copy()
    coord2[0] = random.randint(0, MAX_ELLIPSE)
    coord2[1] = random.randint(0, MAX_ELLIPSE)
    angle = random.randint(0, MAX_ANGLE)
    color = random_color()
    cv2.ellipse(img, (coord1, coord2, angle), color, -1)
    return img


def init_img():
    img = np.full((H, W, C), 0, dtype=np.uint8)
    for i in range(NUM_INIT):
        img = random_rectangle_plot(img)
        img = random_ellipse_plot(img)
    return img


def generate_img(base_img):
    tmp_img = base_img.copy()
    if random.random() < 0.90:
        tmp_img = random_rectangle_plot(tmp_img)
    if random.random() < 0.90:
        tmp_img = random_ellipse_plot(tmp_img)
    return tmp_img


base_img = init_img()
for i in range(NUM_ITERATIONS):
    tmp_imgs = []
    distance = np.linalg.norm(trg_img - base_img)
    tmp_imgs.append((base_img, distance))
    for _ in range(NUM_IMG):
        tmp_img = generate_img(base_img)
        distance = np.linalg.norm(trg_img - tmp_img)
        tmp_imgs.append((tmp_img, distance))
    tmp_imgs.sort(key=lambda x: x[1])
    print("num : ", i, " distance :", tmp_imgs[0][1])
    base_img = tmp_imgs[0][0]
    if (i + 1) % 100 == 0:
        cv2.imwrite(f'result{i+1}.jpg', base_img)