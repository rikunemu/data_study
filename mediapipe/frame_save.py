import cv2
import os
from tqdm import tqdm
import glob
dir_input=r"D:\input_file\Sign_Language\Sign_Language"
dir_output=r"D:\output_file\image\Normal"
len_sign=20
list_class=["test","train"]
def save_all_frames(video_path, dir_path, basename, ext):
    files=glob.glob(video_path+'\*')
    for idx,file in enumerate(files):

        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            return

        os.mkdir(dir_path+r'\video_number'+str(idx+1))
        dir_base_path=dir_path+r'\video_number'+str(idx+1)

        base_path = os.path.join(dir_base_path, basename)

        digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

        n = 0

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('{}_{}_{}.{}'.format(base_path, str(n).zfill(digit),"videonum"+str(idx+1), ext), frame)
                n += 1
            else:
                break
    return
#フォルダ作成
#for i in range(1,21):
    #os.mkdir(r"D:\output_file\csvfile\test\sign_"+str(i))

for i in range(len_sign):
    file_path="sign_"+str(i+1)
    for class_i in tqdm(list_class):
        #count=1
        basefile_inputpath=os.path.join(dir_input,file_path,class_i)
        #ディレクトリの中身分ループ
        #for file_name in tqdm(os.listdir(basefile_inmputpath)):
        basefile_outputpath=os.path.join(dir_output,class_i,file_path)
        save_all_frames(basefile_inputpath,basefile_outputpath,class_i+'_sign_'+str(i+1)+'_frame','jpg')
        #count+=1

