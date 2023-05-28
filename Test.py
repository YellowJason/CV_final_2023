import numpy as np
import cv2
import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()

    seq_path = os.path.join('./ITRI_dataset', args.seq)

    # file of all time stamp
    time_stamp_path = os.path.join(seq_path, 'all_timestamp.txt')
    time_stamp_file = open(time_stamp_path, 'r')

    lines = time_stamp_file.readlines()
    # process img at each time stamp
    for i in range(100): # range(len(lines)):
        line = lines[i]
        print('\r', end='')
        print(f'Reading {i+1}/{len(lines)}', end='')
        # print(line[:-1])
        floder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'
        # read image
        img = cv2.imread(os.path.join(floder_path, 'raw_image.jpg')).astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
        h, w, c = img.shape
        # Filter out white part by hsv
        kernel = np.ones((3,3), np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sensitivity = 100
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.dilate(mask, kernel, iterations = 3)
        res = cv2.bitwise_and(gray, gray, mask=mask)
        # Dilation & Erosion
        res = cv2.dilate(res, kernel, iterations = 2)
        res = cv2.erode(res, kernel, iterations = 2)
        res = cv2.erode(res, kernel, iterations = 2)
        res = cv2.dilate(res, kernel, iterations = 2)
        cv2.imwrite(os.path.join(floder_path, 'image_after_inRange.jpg'), res)
        # Threshold
        gray = cv2.GaussianBlur(gray,(5,5), 0)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        th = cv2.dilate(th, kernel, iterations = 1)
        th = cv2.erode(th, kernel, iterations = 1)
        th = cv2.bitwise_and(th, th, mask=mask)
        cv2.imwrite(os.path.join(floder_path, 'image_after_threshold.jpg'), th)
        # Canny
        canny = cv2.Canny(gray, 30, 140)
        canny = cv2.dilate(canny, kernel, iterations = 1)
        canny = cv2.erode(canny, kernel, iterations = 1)
        # canny = cv2.bitwise_and(canny, canny, mask=mask)
        cv2.imwrite(os.path.join(floder_path, 'image_after_canny.jpg'), canny)

        # read marker
        marker_f = open(os.path.join(floder_path, 'detect_road_marker.csv'), 'r')
        marker = csv.reader(marker_f)
        sift = cv2.xfeatures2d.SIFT_create()
        for row in marker:
            x1, y1 = int(float(row[0])), int(float(row[1]))
            x2, y2 = int(float(row[2])), int(float(row[3]))
            class_id, prob = int(row[4]), float(row[5])
            # print(x1, y1, x2, y2, class_id, prob)

            # Draw marker box
            class_color =  [(255,0,0), (255,255,0), (0,255,0), (0,255,255), (0,0,255)]
            if prob > 0.2:
                cv2.rectangle(img, (x1, y1), (x2, y2), class_color[class_id], 2)

                # find corner point (SIFT)
                '''
                img_box = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                keypoints, des = sift.detectAndCompute(img_box, None)
                for i in keypoints:
                    i.pt = (i.pt[0]+max(0,x1), i.pt[1]+max(0,y1))
                cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=class_color[class_id])
                '''
                # find corner point (Shi-Tomasi), this approach will mark corners on rugged stright line
                img_box = canny[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                corners = cv2.goodFeaturesToTrack(img_box, 20, 0.1, 15) # img. max num of points, quality level, min points distance
                try:
                    corners = np.int0(corners)
                    for i in corners:
                        x,y = i.ravel()
                        cv2.circle(img, (x+max(0,x1), y+max(0,y1)), 3, class_color[class_id], -1)
                except:
                    pass
                
                #
                # contours, hierarchy = cv2.findContours(canny, 3, 2)
                # cnt = contours[0]
                # for i in range(len(contours)):
                #     cnt = contours[i]
                #     approx = cv2.approxPolyDP(cnt, 75, True)
                #     img = cv2.polylines(img, [approx], True, (0, 0, 255), 2)
                

        cv2.imwrite(os.path.join(floder_path, 'image_with_marker.jpg'), img)

if __name__ == '__main__':
    main()