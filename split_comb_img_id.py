import numpy as np

src_file = 'data/NUS-WIDE/database.txt'
img_file = 'data/NUS-WIDE/database_img.txt'
label_file = 'data/NUS-WIDE/database_label.txt'

# ### split
# f_img = open(img_file, 'a')
# f_label = open(label_file, 'a')
# with open(src_file, 'r') as f:
#     for x in f:
#         img = x[0:x.find(' ')]
#         #######
#         img = img.replace('./data/nuswide_81/', '')
#         #######
#         label = x[x.find(' ')+1:]
#         f_img.write(img+'\n')
#         f_label.write(label)
# f_img.close()
# f_label.close()


# ### modify
# f_tgt = open(tgt_file, 'a')
# with open(src_file, 'r') as f:
#     for x in f:
#         # img = x[0:x.find(' ')]
#         #######
#         img = x.replace('./data/nuswide_81/', '')
#         #######
#         # label = x[x.find(' ')+1:]
#         f_tgt.write(img)
#         # f_label.write(label)
# f_tgt.close()

# ### combine
# f_img = open(img_file, 'r')
# f_label = open(label_file, 'r')
# f = open(src_file, 'a')
# for x, y in zip(f_img, f_label):
#     img = x[0:-1]
#     label = y[0:-1]
#     f.write(img+' '+label+'\n')
# f_img.close()
# f_label.close()
# f.close()

# ### combine and onehot
# f_img = open(img_file, 'r')
# f_label = open(label_file, 'r')
# f = open(src_file, 'a')
# for x, y in zip(f_img, f_label):
#     img = x[0:-1]
#     # f.write(img)
#     label = y[0:-1]
#     f.write(img+' '+label+'\n')
#     ##################
#     # target_onehot = np.zeros(10, dtype=int)
#     # target_onehot[int(label)] = 1
#     # for i in target_onehot:
#     #     f.write(' '+str(i))
#     # f.write('\n')
#     ##################
# f_img.close()
# f_label.close()
# f.close()
