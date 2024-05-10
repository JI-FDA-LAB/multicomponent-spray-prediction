import os

dir = './dataset/sample_train'
# dir = './sample_test'

fp = open('./pathTxt/train_path.txt','w+')
# fp = open('./pathTxt/test_img_path.txt','w+')

imgfile_list = os.listdir(dir)
imgfile_list.sort(key= lambda x:int(x[:]))

seqsize = 3
for imgfile in imgfile_list:
    filepath = os.path.join(dir,imgfile)
    img_list = os.listdir(filepath)
    img_list.sort(key= lambda x: int(x[:-4])) # '-4' means removing the file extension like: image123.png --> image123
    #滑窗取序列，步长为8
    for i in range(0, len(img_list)):
       img = img_list[i]
       path = os.path.join(filepath, img)
       fp.write(path+' ')
    fp.write('\n')

fp.close()