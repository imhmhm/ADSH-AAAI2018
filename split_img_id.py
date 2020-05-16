import numpy as np

src_file = 'data/coco/database.txt'
img_file = 'data/coco/database_img.txt'
label_file = 'data/coco/database_label.txt'

f_img = open(img_file, 'a')
f_label = open(label_file, 'a')
with open(src_file, 'r') as f:
    for x in f:
        img = x[0:x.find(' ')]
        label = x[x.find(' ')+1:]
        f_img.write(img+'\n')
        f_label.write(label)
f_img.close()
f_label.close()
