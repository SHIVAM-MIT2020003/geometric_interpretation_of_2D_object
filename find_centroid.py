import cv2

def find_centroid(img, gray_image):
    # img = cv2.imread("images/multi.png", cv2.IMREAD_COLOR)
    # # cv2.imshow("org", img)
    # # cv2.waitKey()
    #
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_image)
    ret,thresh = cv2.threshold(gray_image, 127, 255, 0)

    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)





