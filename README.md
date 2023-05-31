# CV_final_2023
Computer vision final project in NTU, 2023  
Remember to put ITRI_dataset & ITRI_DLC two folder under CV_final_2023/

## Usage
* 找角點:  
    python find_corners.py --seq {seq_num}  
    在每個time stamp資料夾裡存corners.npy，是一個928*1440的boolean array，代表該pixel是否是角點  
  
* 過濾角點:  
    python write_timestamp.py  
    python filter_keypoints.py  
    在每個time stamp資料夾裡存filtered_corners_(y,x).npy，存所有角點的(y,x)座標  
  
## Progress
* 5/30 by 塗兆元  
    write_timestamp:  
    把seq中的all_timestamp.txt切分成四個檔案，測試時比較好用(可只測f camera結果)  
    用法跟Test.py一樣  --seq seqX  
      
    filter_keypoints.py:  
    讀取切分後的四個timestamp，四個for迴圈分別輸出所有過濾雜點影片。  
      
    過濾函數是modify_matrix(A, B):  
    先找B為True的位置，對應A同一位置為中心的(兩倍半徑x兩倍半徑)方格內有沒有True，來決定B這個True要不要保留。  
      
    目前效果不是很好，轉彎時似乎frame之間差異過大，  
    尤其是近處的變化幅度比遠處大，靠近camera的kps會消失(前後frame同一點可能離太遠)  
    直行比較穩定，但近處還是會消失。  
      
    需要測試用作業三的變換矩陣，將前後frame的corners.npy變換到主frame座標後再餵給modify_matrix(A, B)濾。  
      
    我把四個影片分開寫，調整測試的時候可以先註解掉其他三個，測試一個影片的結果就好  

* 5/31 by 黃政勛  
    更新filter_keypoints: 先對前後frame做perspective transform後再篩選標點  
    更新find_corners.py: 用threshold得到的mask，拿來過濾Canny edge結果，減少路面或陰影上的錯誤角點  