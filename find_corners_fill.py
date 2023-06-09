import numpy as np
import cv2
import argparse
import os
import csv
# import cv2.ximgproc as xip

def corner_fill(args):
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
        print(f'Processing {i+1}/{len(lines)}', end='')
        floder_path = os.path.join(seq_path, 'dataset', line[:-1]) # remove '\n'
        # read image
        img = cv2.imread(os.path.join(floder_path, 'raw_image.jpg')).astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('uint8')
        h, w, c = img.shape
        # read camera
        camera_f = open(os.path.join(floder_path, 'camera.csv'), 'r')
        camera = csv.reader(camera_f)
        for c in camera:
            camera = str(c[0])
            camera_mask = cv2.imread(f'ITRI_dataset/camera_info{camera}_mask.png', 0).astype('uint8')
            camera_mask = cv2.dilate(camera_mask, kernel, iterations = 3).astype('bool')
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
        gray = cv2.GaussianBlur(gray,(5,5), 0)
        # gray = xip.jointBilateralFilter(img, gray, 30, 5, 5)
        # cv2.imwrite(os.path.join(floder_path, 'gray_after_jbf.jpg'), gray)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, -20)
        # ret, th = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, kernel, iterations = 2)
        th = cv2.erode(th, kernel, iterations = 2)
        th = cv2.erode(th, kernel, iterations = 2)
        th = cv2.dilate(th, kernel, iterations = 2)
        th = cv2.GaussianBlur(th,(7,7), 0)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_threshold.jpg'), th)
        
        # Canny
        canny = cv2.Canny(gray, 20, 90)
        canny = cv2.dilate(canny, kernel, iterations = 1)
        canny = cv2.erode(canny, kernel, iterations = 1)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_canny.jpg'), canny)
        canny = cv2.bitwise_and(canny, canny, mask=th)
        # cv2.imwrite(os.path.join(floder_path, 'image_after_canny_&_mask.jpg'), canny)

        # read marker
        marker_f = open(os.path.join(floder_path, 'detect_road_marker.csv'), 'r')
        marker = csv.reader(marker_f)
        for row in marker:
            x1, y1 = int(float(row[0])), int(float(row[1]))
            x2, y2 = int(float(row[2])), int(float(row[3]))
            class_id, prob = int(row[4]), float(row[5])
            # if class_id > 2:
            #     continue
            # print(x1, y1, x2, y2, class_id, prob)
            
            # Draw marker box
            class_color =  [(255,0,0), (255,255,0), (0,255,0), (0,255,255), (0,0,255)]
            if prob > 0.2:
                cv2.rectangle(img, (x1, y1), (x2, y2), class_color[class_id], 1)

                img_box = canny[max(0,y1-40):min(h,y2+40), max(0,x1-40):min(w,x2+40)]
                # find corner by findContours & approxPolyDP
                contours, hierarchy = cv2.findContours(img_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                for i in range(len(contours)):
                    cnt = contours[i] # shape: (num, 1, 2)
                    min_d = 10
                    cnt = cnt + np.array([max(0,x1-40), max(0,y1-40)])
                    approx = cv2.approxPolyDP(cnt, 3, True)
                    # print(approx.shape)
                    for j in range(len(approx)):
                        current_point = approx[j][0]
                        next_point = approx[(j + 1) % len(approx)][0]
                        
                        if (current_point[0] > x1+2) and (current_point[1] > y1+2) and (current_point[0] < x2-3) and (current_point[1] < y2-3) and (not camera_mask[current_point[1], current_point[0]]):
                            corner_box = corner_matrix[current_point[1]-min_d:current_point[1]+min_d+1, current_point[0]-min_d:current_point[0]+min_d+1]
                            
                            if (True not in corner_box):
                                img = cv2.circle(img, tuple(current_point), 2, (0,0,255), -1)
                                corner_matrix[current_point[1], current_point[0]] = True

                            # 计算当前点与下一个点之间的距离
                            distance = np.linalg.norm(next_point - current_point)
                            
                            if distance > 1:
                                # 计算点之间的间隔数
                                num_intervals = int(distance)
                                
                                # 计算每个间隔的步长
                                step = (next_point - current_point) / distance
                                
                                for i in range(1, num_intervals):
                                    # 填充点之间的边
                                    fill_point = current_point + i * step
                                    fill_point = fill_point.astype(int)
                                    
                                    if (fill_point[0] > x1+2) and (fill_point[1] > y1+2) and (fill_point[0] < x2-3) and (fill_point[1] < y2-3) and (not camera_mask[fill_point[1], fill_point[0]]):
                                        corner_box = corner_matrix[fill_point[1]-min_d:fill_point[1]+min_d+1, fill_point[0]-min_d:fill_point[0]+min_d+1]
                                        
                                        if (True not in corner_box):
                                            img = cv2.circle(img, tuple(fill_point), 2, (0,0,255), -1)
                                            corner_matrix[fill_point[1], fill_point[0]] = True
                                        
        cv2.imwrite(os.path.join(floder_path, 'image_with_marker.jpg'), img)
        np.save(os.path.join(floder_path, 'corners.npy'), corner_matrix)
        true_coordinates = np.argwhere(corner_matrix)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()
    corner_fill(args)