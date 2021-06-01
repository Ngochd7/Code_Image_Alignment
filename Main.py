import cv2
import numpy as np


def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs

def detectFeatures(img, imgTemplate):
    template_features = getFeatures(imgTemplate)
    template_kps, template_descs = template_features
    # get features from input image
    kps, descs = getFeatures(img)
    # check if keypoints are extracted
    if not kps:
        return None
     # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descs, template_descs, None)
   
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
 
    # Remove not so good matches
    GOOD_MATCH_PERCENT = 0.1
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
 
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = kps[match.queryIdx].pt
        points2[i, :] = template_kps[match.trainIdx].pt

    m, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if m is not None:
        return m
    return None


def detect(test):
    m = detectFeatures(test, imgTemplate)
    if m is not None:
        w = gettemplateImgSize()[1]
        h = gettemplateImgSize()[0]
        # lấy ra hình ảnh logo được phát hiện và xoay nó về góc nhìn chuẩn
        test1 = cv2.warpPerspective(test, m, (w, h))
        return test1
    return test

def gettemplateImgSize():
    return imgTemplate.shape



imgTemplate = cv2.imread('template.jpg')
imgTest = cv2.imread('input.jpg')

imgTest  = detect(imgTest)

# save
path = 'output/result.jpg'
cv2.imwrite(path, imgTest)
# show
cv2.imshow('Result', imgTest)
cv2.waitKey(10000)
cv2.destroyAllWindows()