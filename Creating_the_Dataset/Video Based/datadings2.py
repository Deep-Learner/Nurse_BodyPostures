import os
import glob
import cv2
import pickle as pkl
from tqdm import tqdm
#session= 1
save = True
istrain = True
if istrain:
    trainval = "train"
else:
    trainval = "val"
#"D:\\Dx" #"D:\\D3" #"D:\\Dataset"
#dataset_folder ="D:\\Cam3"
dataset_folder ="D:\\DSTAA"
#dataset_folder ="D:\\Dataset"
for session in range(3,7):
    trainval=f's{session}'
    if save:
        #img_root_dir = "D:\\Dataset" + os.sep + "Crops" + os.sep + trainval + os.sep
        img_root_dir = dataset_folder + os.sep + "Crops" + os.sep + trainval + os.sep
        #img_root_dir = "D:\\Dataset" + os.sep + "Crops" + os.sep + "train" + os.sep
        all_imgs = {}
        for img_path in tqdm(glob.glob(img_root_dir + f"s{session}*.jpg")):
            all_imgs[img_path.split(os.sep)[-1]] = cv2.imread(img_path)
        #all_imgs_file = open("D:\\Dataset" + os.sep + "Crops" + os.sep + trainval + "_all_imgs.pkl", "wb+")
        all_imgs_file = open(dataset_folder + os.sep + "Crops" + os.sep + trainval + "_all_imgs.pkl", "wb+")
        pkl.dump(all_imgs, all_imgs_file)
        all_imgs_file.close()
    else:
        all_imgs_file = open(dataset_folder + os.sep + "Crops" + os.sep + trainval + "_all_imgs.pkl", "rb")
        all_imgs = pkl.load(all_imgs_file)
        all_imgs_file.close()

        for key, value in all_imgs.items():
            cv2.imshow(key, value)
            print(key)
            cv2.waitKey()
