import cv2
import numpy as np
from PIL import Image


def alignImages(img1, img2):
    print("Trying to aligning images...")

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.45

    # Convert images to grayscale
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    cv2.imwrite("result/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    # Use homography
    height, width, channels = img2.shape
    img1AftOffset = cv2.warpPerspective(img1, h, (width, height))

    print("Estimated homography : \n",  h)
    print("Align success.")
    return img1AftOffset, h

def findDifference(img1, img2, maskShow):
    '''
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2  = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret,img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY_INV)
    ret,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite('result/difference/threshold_1.png', img1)
    cv2.imwrite('result/difference/threshold_2.png', img2)
    '''
    for i in [0, 1]:
        # compute difference
        difference = cv2.subtract(img1, img2)
        # color the mask red
        Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
        difference[mask != 255] = [0, 0, 255]
        # add the red mask to the images to make the differences obvious
        img1[mask != 255] = [0, 0, 255]
        img2[mask != 255] = [0, 0, 255]
        maskShow[mask != 255] = 255
        temp = img1
        img1 = img2
        img2 = temp

    # store images
    cv2.imwrite('result/difference/diffOverImage1.png', img1)
    cv2.imwrite('result/difference/diffOverImage2.png', img2)
    cv2.imwrite('result/difference/diffMask.png', difference)
    
    return maskShow

def img_resize_to_same(img, x, y):
    height, width = img.shape[:2]
    black_image = np.zeros((x,y,3), np.uint8)
    black_image[:,:] = (0,0,0)
    image = black_image.copy()
    image[0:height, 0:width] = img.copy()
    return image

def image_aligin(refFilename, offsetImgFilename):
    # Read data.
    referenceImage = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    offsetImage = cv2.imread(offsetImgFilename, cv2.IMREAD_COLOR)
    # resize to same
    bigger_x = max(referenceImage.shape[0], offsetImage.shape[0])
    bigger_y = max(referenceImage.shape[1], offsetImage.shape[1])
    background_img = np.zeros((bigger_x, bigger_y, 3), np.uint8)
    print('image size:{}'.format(background_img.shape))
    referenceImage = img_resize_to_same(referenceImage, bigger_x, bigger_y)
    cv2.imwrite("result/origin1.jpg", referenceImage)
    offsetImage = img_resize_to_same(offsetImage, bigger_x, bigger_y)
    cv2.imwrite("result/origin2.jpg", offsetImage)

    # offsetImage = cv2.resize(offsetImage, (referenceImage.shape[1], referenceImage.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Add outside frame to two images.
    cv2.rectangle(referenceImage,(int(0),int(0)),(int(referenceImage.shape[1]),int(referenceImage.shape[0])),(0,0,255),5)
    cv2.rectangle(offsetImage,(int(0),int(0)),(int(offsetImage.shape[1]),int(offsetImage.shape[0])),(0,255,0),5)

    # Align two images.
    imgOffseted, h = alignImages(offsetImage, referenceImage)
    cv2.imwrite("result/offseted.jpg", imgOffseted)

    # Combine offseted image and reference image
    res = cv2.addWeighted(imgOffseted, 0.5, referenceImage, 0.5, 0)
    cv2.imwrite("result/result.jpg", res)

def image_comparison():
    refFilename1 = 'result/offseted.jpg'
    refFilename2 = 'result/origin1.jpg'
    img1 = cv2.imread(refFilename1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(refFilename2, cv2.IMREAD_COLOR)
    mask = np.zeros((int(img1.shape[0]), int(img1.shape[1])))
    diff = findDifference(img1, img2, mask)
    cv2.imwrite('result/difference/mask.png', mask)


if __name__ == '__main__':
    refFilename = 'data/全景/jpg/0202.jpg'
    offsetImgFilename = 'data/全景/jpg/0313.jpg'
    
    image_aligin()
    image_comparison()

