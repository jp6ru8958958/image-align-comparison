import cv2
import numpy as np

def alignImages(img1, img2):
  print("Trying to aligning images...")

  MAX_FEATURES = 500
  GOOD_MATCH_PERCENT = 0.15

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

def findDifference(img1, img2):

  # compute difference
  difference = cv2.subtract(img1, img2)

  # color the mask red
  Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
  difference[mask != 255] = [0, 0, 255]

  # add the red mask to the images to make the differences obvious
  img1[mask != 255] = [0, 0, 255]
  img2[mask != 255] = [0, 0, 255]

  # store images
  cv2.imwrite('result/difference/diffOverImage1.png', img1)
  cv2.imwrite('result/difference/diffOverImage2.png', img2)
  cv2.imwrite('result/difference/diff.png', difference)
  return 0

if __name__ == '__main__':

  # Read data.
  refFilename = 'data/1016.JPG'
  offsetImgFilename = 'data/1014.JPG'
  referenceImage = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  offsetImage = cv2.imread(offsetImgFilename, cv2.IMREAD_COLOR)

  # Add outside frame to two images.
  cv2.rectangle(referenceImage,(int(0),int(0)),(int(referenceImage.shape[1]),int(referenceImage.shape[0])),(0,0,255),5)
  cv2.rectangle(offsetImage,(int(0),int(0)),(int(offsetImage.shape[1]),int(offsetImage.shape[0])),(0,255,0),5)

  # Align two images.
  imgOffseted, h = alignImages(offsetImage, referenceImage)
  cv2.imwrite("result/offseted.jpg", imgOffseted)

  # Combine offseted image and reference image
  res = cv2.addWeighted(imgOffseted, 0.5, referenceImage, 0.5, 0)
  cv2.imwrite("result/result.jpg", res)

  # Find two images's difference.
  diff = findDifference(imgOffseted, referenceImage)
  # cv2.imwrite("result/difference.jpg", diff)
