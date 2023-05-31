import numpy as np
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description = 'Read dataset & marker')
    parser.add_argument('--seq', default = 'seq1', type=str, help = 'Which sequence do you want to read')
    args = parser.parse_args()

    seq_path = os.path.join('./ITRI_dataset', args.seq)

    # file of diff cam timestamp
    f_timestamp_file = open(os.path.join(seq_path, 'f_timestamp.txt'), 'r')
    lines_f = f_timestamp_file.readlines()
    b_timestamp_file = open(os.path.join(seq_path, 'b_timestamp.txt'), 'r')
    lines_b = b_timestamp_file.readlines()
    fr_timestamp_file = open(os.path.join(seq_path, 'fr_timestamp.txt'), 'r')
    lines_fr = fr_timestamp_file.readlines()
    fl_timestamp_file = open(os.path.join(seq_path, 'fl_timestamp.txt'), 'r')
    lines_fl = fl_timestamp_file.readlines()

    # Make video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter_f_filtered = cv2.VideoWriter(f"{seq_path}/marker_f_filtered.avi", fourcc, 10, (1440, 928))
    videowriter_b_filtered = cv2.VideoWriter(f"{seq_path}/marker_b_filtered.avi", fourcc, 10, (1440, 928))
    videowriter_fr_filtered = cv2.VideoWriter(f"{seq_path}/marker_fr_filtered.avi", fourcc, 10, (1440, 928))
    videowriter_fl_filtered = cv2.VideoWriter(f"{seq_path}/marker_fl_filtered.avi", fourcc, 10, (1440, 928))
 
    files = [lines_f, lines_b, lines_fr, lines_fl]
    name = ['f', 'b', 'fr', 'fl']
    writers = [videowriter_f_filtered, videowriter_b_filtered, videowriter_fr_filtered, videowriter_fl_filtered]
    for j in range(4):
        #fullfill frame 1
        print('\r', end='')
        print(f'{name[j]}_Processing 1/{len(files[j])}', end='')
        frame_1 = os.path.join(seq_path, 'dataset', files[j][0][:-1])
        frame_2 = os.path.join(seq_path, 'dataset', files[j][1][:-1])

        corners_1 = np.load(os.path.join(frame_1, 'corners.npy'))
        corners_2 = np.load(os.path.join(frame_2, 'corners.npy'))

        raw_1 = cv2.imread(os.path.join(frame_1, 'raw_image.jpg')).astype('uint8')
        raw_2 = cv2.imread(os.path.join(frame_2, 'raw_image.jpg')).astype('uint8')

        _, _, corners_2_to_1 = img_transform(raw_2, raw_1, corners_2)
        filtered_1 = modify_matrix(corners_2_to_1, corners_1)
        img_video_1 = raw_1
        true_coordinates_1 = np.argwhere(filtered_1)
        np.save(os.path.join(frame_1, 'filtered_corners_(y,x).npy'), true_coordinates_1)
        true_coordinates_tuples_1 = [tuple(coordinate) for coordinate in true_coordinates_1]
        for coordinate in true_coordinates_tuples_1:
            swap = (coordinate[1], coordinate[0])
            cv2.circle(img_video_1, swap, 2, (0, 255, 255), -1)
        writers[j].write(img_video_1)

        # for loop frame 2 to last2    
        for i in range(1, len(files[j]) - 1, 1):
            frame_now = os.path.join(seq_path, 'dataset', files[j][i][:-1])
            frame_former1 = os.path.join(seq_path, 'dataset', files[j][i - 1][:-1])
            frame_later1 = os.path.join(seq_path, 'dataset', files[j][i + 1][:-1])
            print('\r', end='')
            print(f'{name[j]}_Processing {i+1}/{len(files[j])}', end='')

            corners_n = np.load(os.path.join(frame_now, 'corners.npy'))
            corners_f1 = np.load(os.path.join(frame_former1, 'corners.npy'))
            corners_l1 = np.load(os.path.join(frame_later1, 'corners.npy'))
            # use raw data for perspective transform
            raw_n = cv2.imread(os.path.join(frame_now, 'raw_image.jpg')).astype('uint8')
            raw_f1 = cv2.imread(os.path.join(frame_former1, 'raw_image.jpg')).astype('uint8')
            raw_l1 = cv2.imread(os.path.join(frame_later1, 'raw_image.jpg')).astype('uint8')
            H_f1_to_n, img_f1_to_n, corners_f1_to_n = img_transform(raw_f1, raw_n, corners_f1)
            # cv2.imwrite(os.path.join(frame_now, 'image_after_forward_f1.jpg'), img_f1_to_n)
            H_l1_to_n, img_l1_to_n, corners_l1_to_n = img_transform(raw_l1, raw_n, corners_l1)
            # cv2.imwrite(os.path.join(frame_now, 'image_after_forward_l1.jpg'), img_l1_to_n)
            
            out = modify_matrix(np.logical_or(corners_f1_to_n, corners_l1_to_n), corners_n)

            img = cv2.imread(os.path.join(frame_now, 'raw_image.jpg')).astype('uint8')
            img_video = cv2.imread(os.path.join(frame_now, 'raw_image.jpg')).astype('uint8')
            # plot original corners
            true_coordinates = np.argwhere(corners_n)
            true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
            for coordinate in true_coordinates_tuples:
                swap = (coordinate[1], coordinate[0])
                cv2.circle(img, swap, 4, (0, 0, 255), -1)
                cv2.circle(img_video, swap, 4, (0, 0, 255), -1)
            # Plot filtered corners
            true_coordinates = np.argwhere(out)
            np.save(os.path.join(frame_now, 'filtered_corners_(y,x).npy'), true_coordinates)
            true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
            for coordinate in true_coordinates_tuples:
                swap = (coordinate[1], coordinate[0])
                cv2.circle(img, swap, 2, (0, 255, 255), -1)
                cv2.circle(img_video, swap, 2, (0, 255, 255), -1)
            # plot corners of former frame
            true_coordinates = np.argwhere(corners_f1_to_n)
            true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
            for coordinate in true_coordinates_tuples:
                swap = (coordinate[1], coordinate[0])
                cv2.circle(img, swap, 2, (0, 255, 0), -1)
            # plot corners of later frame
            true_coordinates = np.argwhere(corners_l1_to_n)
            true_coordinates_tuples = [tuple(coordinate) for coordinate in true_coordinates]
            for coordinate in true_coordinates_tuples:
                swap = (coordinate[1], coordinate[0])
                cv2.circle(img, swap, 2, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(frame_now, 'image_3_corners.jpg'), img)
            writers[j].write(img_video)

        #fullfill frame Last1
        print('\r', end='')
        print(f'{name[j]}_Processing {len(files[j])}/{len(files[j])}', end='')
        frame_last1 = os.path.join(seq_path, 'dataset', files[j][-1][:-1])
        frame_last2 = os.path.join(seq_path, 'dataset', files[j][-2][:-1])

        corners_last1 = np.load(os.path.join(frame_last1, 'corners.npy'))
        corners_last2 = np.load(os.path.join(frame_last2, 'corners.npy'))

        raw_last1 = cv2.imread(os.path.join(frame_last1, 'raw_image.jpg')).astype('uint8')
        raw_last2 = cv2.imread(os.path.join(frame_last2, 'raw_image.jpg')).astype('uint8')

        _, _, corners_L2_to_L1 = img_transform(raw_last2, raw_last1, corners_last2)
        filtered_last1 = modify_matrix(corners_L2_to_L1, corners_last1)
        img_video_last1 = raw_last1
        true_coordinates_last1 = np.argwhere(filtered_last1)
        np.save(os.path.join(frame_last1, 'filtered_corners_(y,x).npy'), true_coordinates_last1)
        true_coordinates_tuples_last1 = [tuple(coordinate) for coordinate in true_coordinates_last1]
        for coordinate in true_coordinates_tuples_last1:
            swap = (coordinate[1], coordinate[0])
            cv2.circle(img_video_last1, swap, 2, (0, 255, 255), -1)
        writers[j].write(img_video_last1)

        print('')
        writers[j].release()

def modify_matrix(A, B):
    rows, cols = A.shape
    result = np.copy(B)
    indices = np.argwhere(B)  # 找到B矩陣中所有為True的位置
    radius = 20
    for idx in indices:
        i, j = idx[0], idx[1]
        top_i = max(i-radius, 0)
        left_j = max(j-radius, 0)
        bot_i = min(rows, i+radius+1)
        right_j = min(cols, j+radius+1)
        neighborhood = A[top_i:bot_i, left_j:right_j]  # 取得中心41*41方格

        if np.any(neighborhood):
            result[i, j] = True
        else:
            result[i, j] = False
    
    return result

# perspective transform from img1 to img2
def img_transform(img1, img2, corners1):
    MIN_MATCH_COUNT = 10
    h, w, c = img1.shape

    # Initiate detector
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        matchesMask = mask.ravel().tolist()
        h,w,c = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,H)
        
        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    ##################
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                matchesMask = matchesMask, # draw only inliers
    #                flags = 2)
    # img_match_points = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    ##################
    
    out_img = np.zeros((h,w,c))
    out_corners = np.zeros((h,w)).astype(bool)
    xc, yc = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1), sparse = False)
    xrow = xc.reshape((w*h, 1, 1))
    yrow = yc.reshape((w*h, 1, 1))
    pts = np.concatenate((xrow, yrow), axis = 2).astype(float)
    dst = cv2.perspectiveTransform(pts, H)
    
    dsty = np.round(dst[:, :, 1].reshape(h, w)).astype(int)
    dstx = np.round(dst[:, :, 0].reshape(h, w)).astype(int)
    
    h_mask = (0<=dsty)*(dsty<h)
    w_mask = (0<=dstx)*(dstx<w)
    mask   = h_mask*w_mask
    out_img[dsty[mask], dstx[mask]] = img1[yc[mask], xc[mask]]
    out_corners[dsty[mask], dstx[mask]] = corners1[yc[mask], xc[mask]]

    return H, out_img, out_corners


if __name__ == '__main__':
    main()