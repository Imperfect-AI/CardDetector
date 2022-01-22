############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
from card_detector import Cards
import os


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

# Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Define font to use
# font = cv2.FONT_HERSHEY_SIMPLEX

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + "/Card_Imgs/ggpoker/2022_01_22_isolate/")
train_suits = Cards.load_suits(path + "/Card_Imgs/ggpoker/2022_01_22_isolate/")


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.


# Begin capturing frames

# Grab frame from video stream
img_path = "D:\\workspace\\GGPoker\\PokerUI\\card_detector\\OCR\\1.jpg"
# x: 278:718, y: 315:425
ori_image = cv2.imread(img_path)

# BoardCards
def ParseBoardCard(ori_image):
    image = ori_image[310:425, 270:718]

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
    # cv2.imshow("pre_pro", pre_proc)
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)
    # print(cnts_sort)
    # If there are no contours, do nothing
    cards = []
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                print("detect one card, try to analyze it.")
                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                cards.append(Cards.preprocess_card(cnts_sort[i], image, i))

                # Find the best rank and suit match for the card.
                (
                    cards[k].best_rank_match,
                    cards[k].best_suit_match,
                    cards[k].rank_diff,
                    cards[k].suit_diff,
                ) = Cards.match_card(cards[k], train_ranks, train_suits)

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k = k + 1

        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if len(cards) != 0:
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.
    # cv2.putText(image, "FPS: "+str(int(frame_rate_calc)),
    #            (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Finally, display the image with the identified cards!
    # cv2.imshow("Card Detector", image)
    # cv2.imwrite("/home/sqliu/img.png", image)

    # Pause
    # key = cv2.waitKey(0) & 0xFF
    # if key == ord("c"):
    #    cv2.destroyAllWindows()

    # Close all windows and close the PiCamera video stream.
    # cv2.destroyAllWindows()
    return cards


card_hight = 67
card_width = 48


def ParseHandCard1(ori_image):
    # ParseCard(ori_image, 420, 474, 564, 635)
    image_info = ori_image.shape
    matRotate = cv2.getRotationMatrix2D((image_info[0], image_info[1]), -6, 1)
    rotate_img = cv2.warpAffine(ori_image, matRotate, (image_info[0], image_info[1]))
    # cv2.imwrite(
    #    "D:\\workspace\\GGPoker\\PokerUI\\card_detector\\OCR\\rotate_1_1.jpeg",
    #    rotate_img,
    # )
    # cv2.imshow("ParseHandCard1: ", rotate_img)
    # key = cv2.waitKey(0) & 0xFF
    return ParseCard(rotate_img, 341, 316)


def ParseHandCard2(ori_image):
    # ParseCard(ori_image, 420, 474, 564, 635)
    # rotate
    image_info = ori_image.shape
    matRotate = cv2.getRotationMatrix2D((image_info[0], image_info[1]), 6, 1)
    rotate_img = cv2.warpAffine(ori_image, matRotate, (image_info[0], image_info[1]))
    # cv2.imwrite(
    #    "D:\\workspace\\GGPoker\\PokerUI\\card_detector\\OCR\\rotate_1_2.jpeg",
    #    rotate_img,
    # )
    # cv2.imshow("ParseHandCard2: ", rotate_img)
    # key = cv2.waitKey(0) & 0xFF
    return ParseCard(rotate_img, 374, 291)


def ParseCard(ori_image, x1, y1):
    x2 = x1 + card_hight
    y2 = y1 + card_width
    Qcorner = ori_image[x1:x2, y1:y2]
    # this image is already the corner of the card. we need to resize it and grey it.

    # cv2.imshow("ParseCardCorner: ", Qcorner)
    # key = cv2.waitKey(0) & 0xFF
    card = Cards.preprocess_corner(Qcorner)
    (
        best_rank_match,
        best_suit_match,
        rank_diff,
        suit_diff,
    ) = Cards.match_card(card, train_ranks, train_suits)
    print("Rank: ", best_rank_match, ",suit: ", best_suit_match)
    return (best_rank_match, best_suit_match)


if __name__ == "__main__":
    cv2.imshow("ori_image", ori_image)
    ParseHandCard1(ori_image)
    ParseHandCard2(ori_image)
    # board_cards = ParseBoardCard(ori_image)
    # for board_card in board_cards:
    #    print(
    #        "Board card: ",
    #        board_card.best_rank_match,
    #        ", suit: ",
    #        board_card.best_suit_match,
    #    )
    key = cv2.waitKey(0) & 0xFF
    if key == ord("c"):
        cv2.destroyAllWindows()
