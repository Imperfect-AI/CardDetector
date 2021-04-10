import glob
import sys
import base64
import cv2

sys.path.append("/home/liushiqi9/workspace/slumbot2019/src/thrift/gen-py")
sys.path.append("/home/liushiqi9/workspace/thrift-0.13.0/lib/py/build/lib.linux-x86_64-3.8/")

from recognition_server import RecognitionServer
from recognition_server.ttypes import *

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.protocol import TJSONProtocol
from thrift.server import TServer, THttpServer
import http.server as BaseHTTPServer

import CardDetector

from PIL import Image
import cv2
from io import StringIO
import numpy as np


def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


class RecognitionserverHandler:
    def __init__(self):
        self.log = {}

    def AnalyzeCard(self, image):
        """
      Parameters:
       - image

      """
        print("Recv analyze card")
        # decoded_image = readb64(image)
        nparr = np.frombuffer(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        print("Recv data: ", base64.b64encode(image))
        # cv2.imshow("decoded_image", img_np)

        width = 992
        height = 762
        dim = (width, height)

        # resize image
        resized = cv2.resize(img_np, dim, interpolation=cv2.INTER_AREA)

        (hand_card_1, hand_card_suit_1) = CardDetector.ParseHandCard1(resized)
        # key = cv2.waitKey(0) & 0xFF
        (hand_card_2, hand_card_suit_2) = CardDetector.ParseHandCard2(resized)
        board_cards = CardDetector.ParseBoardCard(resized)

        print("Hand: ", hand_card_1, ", hand suit: ", hand_card_suit_1)
        print("Hand: ", hand_card_2, ", hand suit: ", hand_card_suit_2)
        print("Board: ", board_cards)

        result = RecognitionResult()
        result.hole_cards = []
        result.board_cards = []
        result.hole_cards.append(Card(hand_card_1, hand_card_suit_1))
        result.hole_cards.append(Card(hand_card_2, hand_card_suit_2))
        for board_card in board_cards:
            result.board_cards.append(Card(board_card.best_rank_match, board_card.best_suit_match))

        # cv2.destroyAllWindows()
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord("c"):
        #    cv2.destroyAllWindows()
        return result


if __name__ == "__main__":
    handler = RecognitionserverHandler()
    processor = RecognitionServer.Processor(handler)
    transport = TSocket.TServerSocket(host="0.0.0.0", port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(
    #     processor, transport, tfactory, pfactory)

    print("Starting the server...")
    server.serve()
    print("done.")

