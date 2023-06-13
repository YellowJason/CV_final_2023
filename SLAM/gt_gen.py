with open('../ITRI_dataset/seq1/localization_timestamp.txt', 'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]

with open('gt_pose.txt', 'w+') as gt:
    for i in range(len(lines)):
        print(lines[i])
        with open("../ITRI_dataset/seq1/dataset/%s/gound_turth_pose.csv" %(lines[i]), "r") as file:
            lines2 = file.readlines()
        print(lines2)
        line2 = [line.strip() for line in lines2]
        gt.write(line2[0])
        gt.write("\n")
    gt.close()
