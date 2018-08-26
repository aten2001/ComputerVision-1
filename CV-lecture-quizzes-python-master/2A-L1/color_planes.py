import cv2

# Color planes

img = cv2.imread('images/fruit.png')
cv2.imshow("Fruit image", img)

print img.shape

# TODO: Select a color plane, display it, inspect values from a row

img_red = img[:, :, 2]
img_green = img[:, :, 1]
img_blue = img[:, :, 0]

cv2.imshow("Red Color Plane Image", img_red)

# select a color, pick a row and print the value.
print 'Green color plan 100th row \n {}'.format(img_green[99, :])


# Let's draw a horizontal line across that row we picked, just for fun
# But in order to see the line in color we need to convert green to a 3-channel array
green_bgr = cv2.cvtColor(img_green, cv2.COLOR_GRAY2BGR)

# You will notice that cv2.line uses x-y coordinates instead of row-cols
cv2.line(green_bgr, (0, 99), (img_green.shape[1], 99), (0, 0, 255))
cv2.imshow("50-th row drawn on the green color plane", green_bgr)



cv2.waitKey(0)
