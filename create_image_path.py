import os

# For every training set and test set, create their path,
# which will be used in LSTM_training.py and predict.py

dir = './dataset/sample_train/extrapolation/octane-fixed'
fp = open('./path/octane-fixed-extra-train-path.txt','w+')

# dir = './dataset/sample_test/extrapolation/octane-fixed'
# fp = open('./path/octane-fixed-extra-test-path.txt','w+')

# dir = './dataset/sample_test/interpolation/AEC-test-set-threshold'
# fp = open('./path/AEC-test-path.txt','w+')
imgfile_list = os.listdir(dir)
imgfile_list.sort(key= lambda x:int(x[:]))

for imgfile in imgfile_list:
    filepath = os.path.join(dir,imgfile)
    img_list = os.listdir(filepath)
    img_list.sort(key= lambda x: int(x[:-4])) # '-4' means removing the file extension like: image123.png --> image123

    for i in range(0, len(img_list)):
       img = img_list[i]
       path = os.path.join(filepath, img)
       fp.write(path+' ')
    fp.write('\n')

fp.close()