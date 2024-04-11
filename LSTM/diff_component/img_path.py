import os

dir = './sorted_test'
fp = open('./test_img_path.txt','w+')
imgfile_list = os.listdir('./sorted_test')
imgfile_list.sort(key= lambda x:int(x[:]))
# print(imgfile_list)
seqsize = 5
for imgfile in imgfile_list:
    filepath = os.path.join(dir,imgfile)
    img_list = os.listdir(filepath)
    img_list.sort(key= lambda x: int(x[:-4])) # '-4' means removing the file extension like: image123.png --> image123
    #滑窗取序列，步长为8
    for i in range(0, len(img_list)): # Originally, 16 should be 8, then there was an overlap of (seqsize -8) images
        # between consequtive sequences, which is to help capture temporal continuity across sequences.
        # In our case, there's no need for continuity across sequences.
       img = img_list[i]
       path = os.path.join(filepath, img)
       fp.write(path+' ')
    fp.write('\n')

fp.close()