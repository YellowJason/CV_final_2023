import numpy as np
import cv2
import argparse
import os
import csv
import cv2.ximgproc as xip

def find_corners(args):
    if args.seq in ['seq1', 'seq2', 'seq3']:
        seq_path = os.path.join('./ITRI_dataset', args.seq)
    elif args.seq in ['test1', 'test2']:
        seq_path = os.path.join('./ITRI_DLC', args.seq)
    # file of all time stamp
    time_stamp_path = os.path.join(seq_path, 'all_timestamp.txt')
    time_stamp_file = open(time_stamp_path, 'r')
    lines = time_stamp_file.readlines()

    # Make video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter_f = cv2.VideoWriter(f"{seq_path}/marker_f.avi", fourcc, 10, (1440, 928))
    videowriter_b = cv2.VideoWriter(f"{seq_path}/marker_b.avi", fourcc, 10, (1440, 928))
    videowriter_fr = cv2.VideoWriter(f"{seq_path}/marker_fr.avi", fourcc, 10, (1440, 928))
    videowriter_fl = cv2.VideoWriter(f"{seq_path}/marker_fl.avi", fourcc, 10, (1440, 928))
    # process img at each time stamp
    kernel = np.ones((3,3), np.uint8)
    for i in range(len(lines)):
        line = lines[i] 
        print('\r', end='')
        print(f'Finding corners {i+1}/{len(lines)}', end='')
        floder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'
        # read image
        img = cv2.imread(os.path.join(floder_path, 'raw_image.jpg')).astype('uint8')
        img = cv2.bilateralFilter(img, 15, 100, 20)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_bf.jpg'), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
        h, w, c = img.shape
        # read camera
        camera_f = open(os.path.join(floder_path, 'camera.csv'), 'r')
        camera = csv.reader(camera_f)
        for c in camera:
            camera = str(c[0])
            camera_mask = cv2.imread(f'ITRI_dataset/camera_info{camera}_mask.png', 0).astype('uint8')
            camera_mask = cv2.dilate(camera_mask, kernel, iterations = 5).astype('bool')
            camera = camera.split('_')[4] # fl, f, b, fr
            # print(camera)
        # Record corner position
        corner_matrix = np.zeros((h,w)).astype(bool)
        # Filter out white part by hsv
        '''
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sensitivity = 120
        lower_white = np.array([0,0,255-sensitivity])
        upper_white = np.array([255,sensitivity,255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.dilate(mask, kernel, iterations = 3)
        cv2.imwrite(os.path.join(floder_path, 'mask.jpg'), mask)
        '''
        # Filter out white part by threshold
        # gray = cv2.GaussianBlur(gray,(3,3), 0)
        # gray = cv2.bilateralFilter(gray, 15, 20, 20)
        # gray = xip.jointBilateralFilter(img, gray, 30, 5, 5)
        # cv2.imwrite(os.path.join(floder_path, 'gray_after_jbf.jpg'), gray)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 121, -10)
        th = cv2.dilate(th, kernel, iterations = 2)
        th = cv2.erode(th, kernel, iterations = 2)
        th = cv2.erode(th, kernel, iterations = 2)
        th = cv2.dilate(th, kernel, iterations = 6)
        # th = cv2.GaussianBlur(th,(7,7), 0)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_threshold.jpg'), th)
        
        # Canny
        canny = cv2.Canny(gray, 15, 90)
        canny = cv2.dilate(canny, kernel, iterations = 1)
        canny = cv2.erode(canny, kernel, iterations = 1)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_canny.jpg'), canny)
        # canny = cv2.bitwise_and(canny, canny, mask=th)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_canny_&_mask.jpg'), canny)
        # Dectect corners
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cnt = contours[i] # shape: (num, 1, 2)
            min_d = 5
            approx = cv2.approxPolyDP(cnt, 5, True)
            # print(approx.shape)
            img = cv2.polylines(img, [approx], True, (0, 255, 255), 1)
            for j in approx:
                # delete points on box boundary, repeated, and on edge of mask
                if not camera_mask[j[0][1], j[0][0]]:
                    img = cv2.circle(img, tuple(j[0]), 2, (0,0,255), -1)
                    corner_matrix[j[0][1], j[0][0]] = True
        # cv2.imwrite(os.path.join(floder_path, 'find_corner_whole.jpg'), img)
        
        # read marker
        marker_f = open(os.path.join(floder_path, 'detect_road_marker.csv'), 'r')
        marker = csv.reader(marker_f)
        selected = np.zeros((h,w)).astype(bool)
        for row in marker:
            x1, y1 = int(float(row[0])), int(float(row[1]))
            x2, y2 = int(float(row[2])), int(float(row[3]))
            class_id, prob = int(row[4]), float(row[5])
            if class_id > 2:
                continue
            # print(x1, y1, x2, y2, class_id, prob)
            
            # Draw marker box
            class_color =  [(255,0,0), (255,255,0), (0,255,0), (0,255,255), (0,0,255)]
            if prob > 0.2:
                y1, y2, x1, x2 = max(0,y1), min(h,y2), max(0,x1), min(w,x2)
                cv2.rectangle(img, (x1, y1), (x2, y2), class_color[class_id], 1)
                selected[y1:y2, x1:x2] = np.ones((y2-y1, x2-x1)).astype(bool)

        # cv2.imwrite(os.path.join(floder_path, 'select.jpg'), selected*255)
        corner_matrix = np.logical_and(corner_matrix, selected)
        true_coordinates = np.argwhere(corner_matrix)
        for co in true_coordinates:
            # print(co[1], co[0])
            img = cv2.circle(img, (co[1], co[0]), 2, (255,0,0), -1)
        cv2.imwrite(os.path.join(floder_path, 'image_with_marker.jpg'), img)
        np.save(os.path.join(floder_path, 'corners.npy'), corner_matrix)
        np.save(os.path.join(floder_path, 'corners_(y,x).npy'), true_coordinates)
        # Write video
        if camera == 'f':
            videowriter_f.write(img)
        elif camera == 'b':
            videowriter_b.write(img)
        elif camera == 'fr':
            videowriter_fr.write(img)
        elif camera == 'fl':
            videowriter_fl.write(img)

    videowriter_f.release()
    videowriter_b.release()
    videowriter_fr.release()
    videowriter_fl.release()
    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()
    find_corners(args)