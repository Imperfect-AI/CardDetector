import cv2
import glob
import base64
import sys
import os
import numpy as np


sys.path.append("/home/liushiqi9/workspace/slumbot2019/src/thrift/gen-py")
sys.path.append("/home/liushiqi9/workspace/thrift-0.13.0/lib/py/build/lib.linux-x86_64-3.8/")


from recognition_server import RecognitionServer
from recognition_server.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


# Make socket
transport = TSocket.TSocket("127.0.0.1", 9090)

# Buffering is critical. Raw sockets are very slow
transport = TTransport.TBufferedTransport(transport)

# Wrap in a protocol
protocol = TBinaryProtocol.TBinaryProtocol(transport)

# Create a client to use the protocol encoder
client = RecognitionServer.Client(protocol)

# Connect!
transport.open()

img_path = "/home/liushiqi9/workspace/OpenCV-Playing-Card-Detector/wrong_img/w1.jpeg"
ori_image = cv2.imread(img_path)
img = base64.b64encode(ori_image)

img_str = open(img_path, "rb").read()
quotient = client.AnalyzeCard(bytearray(img_str))

for hand_card in quotient.hole_cards:
    print("HandCard: ", hand_card.rank, ", suit: ", hand_card.suit)
for board_card in quotient.board_cards:
    print("BoardCard: ", board_card.rank, ", suit: ", board_card.suit)

transport.close()
