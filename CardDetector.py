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
import Cards
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
train_ranks = Cards.load_ranks(path + "/Card_Imgs/ggpoker/new_isolate/")
train_suits = Cards.load_suits(path + "/Card_Imgs/ggpoker/new_isolate/")


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.


# Begin capturing frames

# Grab frame from video stream
img_path = "/home/yjb/poker_ai/OpenCV-Playing-Card-Detector/OCR/handcard.jpeg"
# x: 278:718, y: 315:425
ori_image = cv2.imread(img_path)

# BoardCards
def ParseBoardCard(ori_image):
    image = ori_image[310:425, 270:718]

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
    cv2.imshow("pre_pro", pre_proc)
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)
    # print(cnts_sort)
    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
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
    cv2.imwrite("/home/yjb/img.png", image)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("c"):
        cv2.destroyAllWindows()
    # Close all windows and close the PiCamera video stream.
    # cv2.destroyAllWindows()


def ParseHandCard1(ori_image):
    # ParseHandCard(ori_image, 420, 474, 564, 635)
    ParseHandCard(ori_image, 564, 640, 420, 474)


def ParseHandCard2(ori_image):
    # ParseHandCard(ori_image, 420, 474, 564, 635)
    ParseHandCard(ori_image, 561, 635, 475, 527)


def ParseHandCard(ori_image, x1, x2, y1, y2):
    Qcorner = ori_image[x1:x2, y1:y2]
    # this image is already the corner of the card. we need to resize it and grey it.

    card = Cards.preprocess_corner(Qcorner)
    (best_rank_match, best_suit_match, rank_diff, suit_diff,) = Cards.match_card(
        card, train_ranks, train_suits
    )
    print("Rank: ", best_rank_match, ",suit: ", best_suit_match)


cv2.imshow("ori_image", ori_image)
ParseHandCard1(ori_image)
ParseHandCard2(ori_image)
key = cv2.waitKey(0) & 0xFF
if key == ord("c"):
    cv2.destroyAllWindows()

