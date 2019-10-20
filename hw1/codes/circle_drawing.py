import cv2
import numpy as np
import math

img = ""
# mouse callback function
def draw_circle(event, x, y, flags, param):
    global img
    # global x1, y1, radius, num
    # if event == cv2.EVENT_LBUTTONDOWN:
        # x1, y1 = x, y

    if event == cv2.EVENT_LBUTTONDBLCLK:
        # num += 1
        # radius = int(math.hypot(x - x1, y - y1))
        cv2.circle(img, (x,y), 12, (255, 0,), 1)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, f'label: {num}', (x + 30, y + 30), font, 1, (200, 255, 155), 1, cv2.LINE_AA)


def xx():
    global img
    num = 0
    # Create a black image and a window
    windowName = 'Drawing'
    img = cv2.imread('../saved_images/img_0.png', cv2.IMREAD_COLOR)
    cv2.namedWindow(windowName)
    # bind the callback function to window
    cv2.setMouseCallback(windowName, draw_circle)
    while (True):
        cv2.imshow(windowName, img)
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    xx()
