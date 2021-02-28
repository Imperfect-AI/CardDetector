### Takes a card picture and creates a top-down 200x300 flattened image
### of it. Isolates the suit and rank and saves the isolated images.
### Runs through A - K ranks and then the 4 suits.

# Import necessary packages
import cv2
import numpy as np
import time
import Cards
import os

img_path = (
    os.path.dirname(os.path.abspath(__file__)) + "/Card_Imgs/ggpoker/new_isolate/"
)

print(img_path)
IM_WIDTH = 1280
IM_HEIGHT = 720

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# Use counter variable to switch from isolating Rank to isolating Suit
i = 0
image_dir = "/home/yjb/poker_ai/OpenCV-Playing-Card-Detector/Card_Imgs/ggpoker/"
name_list = [
    "Ace",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Spades",
    "Diamonds",
    "Clubs",
    "Hearts",
]
for index in range(17):
    i = i + 1
    Name = name_list[i - 1]
    source_file_name = "original_img/" + str(i) + ".png"
    filename = Name + ".jpg"

    print("start processing imgage: ", source_file_name)
    image = cv2.imread(image_dir + source_file_name)
    cv2.imshow(source_file_name, image)
    # Pre-process image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    retval, thresh = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    # Find contours and sort them by size
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Assume largest contour is the card. If there are no contours, print an error
    flag = 0
    image2 = image.copy()

    if len(cnts) == 0:
        print("No contours found!")
        quit()

    card = cnts[0]

    # Approximate the corner points of the card
    peri = cv2.arcLength(card, True)
    approx = cv2.approxPolyDP(card, 0.01 * peri, True)
    pts = np.float32(approx)

    x, y, w, h = cv2.boundingRect(card)

    # Flatten the card and convert it to 200x300
    print("pts: ", pts)
    print("w: ", w)
    print("h: ", h)
    warp = Cards.flattener(image, pts, w, h)

    x = 0
    y = 0
    w = 115
    h = 85
    w2 = 75
    color = (255, 255, 0)
    cv2.rectangle(warp, (x, y), (x + h, y + w), color, thickness=1)
    cv2.rectangle(warp, (x, y + w), (x + h, y + w + w2), color, thickness=1)

    # Grab corner of card image, zoom, and threshold
    cv2.imshow("Warp" + str(i), warp)

    rank_corner = warp[0:115, 0:85]
    # key = cv2.waitKey(0) & 0xFF
    rank_corner_zoom = cv2.resize(rank_corner, (0, 0), fx=4, fy=4)
    rank_corner_blur = cv2.GaussianBlur(rank_corner_zoom, (5, 5), 0)
    retval, rank_corner_thresh = cv2.threshold(
        rank_corner_blur, 155, 255, cv2.THRESH_BINARY_INV
    )
    # cv2.imshow("Rank_corner" + str(i), rank_corner)
    # cv2.imshow("Rank_cornerZoom" + str(i), rank_corner_zoom)
    # cv2.imshow("Rank_cornerThresh" + str(i), rank_corner_thresh)

    suit_corner = warp[115:190, 0:85]
    # key = cv2.waitKey(0) & 0xFF
    suit_corner_zoom = cv2.resize(suit_corner, (0, 0), fx=4, fy=4)
    suit_corner_blur = cv2.GaussianBlur(suit_corner_zoom, (5, 5), 0)
    retval, suit_corner_thresh = cv2.threshold(
        suit_corner_blur, 155, 255, cv2.THRESH_BINARY_INV
    )
    # cv2.imshow("Suit_corner" + str(i), suit_corner)
    # cv2.imshow("Suit_cornerZoom" + str(i), suit_corner_zoom)
    # cv2.imshow("Suit_cornerThresh" + str(i), suit_corner_thresh)
    # continue

    # Isolate suit or rank
    if i <= 13:  # Isolate rank
        # rank = corner_thresh[20:185, 0:128] # Grabs portion of image that shows rank
        rank = rank_corner_thresh
        rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rank_cnts = sorted(rank_cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(rank_cnts[0])
        rank_roi = rank[y : y + h, x : x + w]
        rank_sized = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        final_img = rank_sized

    if i > 13:  # Isolate suit
        suit = suit_corner_thresh
        cv2.imshow("Suit" + str(i), suit)
        suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        suit_cnts = sorted(suit_cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(suit_cnts[0])
        suit_roi = suit[y : y + h, x : x + w]
        suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        final_img = suit_sized

    cv2.imshow("Image", final_img)
    cv2.imwrite(img_path + filename, final_img)
    print("write into dir: ", img_path + filename)


print('Press "c" to continue.')
key = cv2.waitKey(0) & 0xFF
if key == ord("c"):
    cv2.destroyAllWindows()
# camera.close()
