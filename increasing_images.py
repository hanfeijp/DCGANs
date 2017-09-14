from PIL import Image
import os
import _pickle as cPickle
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt



def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


path1="/Users/hagiharatatsuya/Desktop/ロビーダミー128px"
images = os.listdir(path1) #ディレクトリのパスをここに書く
namedic = {int(name.split(".")[0]):name for name in images}
name_order=[]
for lst in sorted(namedic.items()):
    name_order.append(lst[1])
len(name_order)



# In[76]: set images in numerical order


img_batch=[]
for i in name_order:
    img_batch.append(cv2.imread(path1+'/'+i))
plt.imshow(img_batch[111], 'gray')#32×32
plt.show() 
len(img_batch)


# In[77]: # contrast


min_table = 10
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
LUT_LC = np.arange(256, dtype = 'uint8' )

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255

# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

# 変換

high_cont_img = []
for i in img_batch:
    high_cont_img.append(cv2.LUT(i, LUT_HC))
low_cont_img = []
for i in img_batch:
    low_cont_img.append(cv2.LUT(i, LUT_LC))


# In[78]: ganmma, ganmma2


gamma = 1.5
gamma1 = 0.75
look_up_table = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
ganma=[]
for i in img_batch:
    ganma.append(cv2.LUT(i, look_up_table))


look_up_table2 = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table2[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma1)
ganma1=[]
for i in img_batch:
    ganma1.append(cv2.LUT(i, look_up_table2))


# In[79]:


average_square = (2,2)
blur_img=[]
for i in img_batch:
    blur_img.append(cv2.blur(i, average_square))


# In[80]: sharp


# シャープの度合い
k = 0.3
# 粉雪（シャープ化）
shape_operator = np.array([[0, -k, 0],[-k, 1 + 4 * k, -k],[0,-k, 0]])
 
img_tmp=[]
    # 作成したオペレータを基にシャープ化
for i in img_batch:
    img_tmp.append(cv2.convertScaleAbs(cv2.filter2D(i, -1, shape_operator))) 



# In[81]: sharp2


# シャープの度合い
k = 0.1
# 粉雪（シャープ化）
shape_operator = np.array([[0, -k, 0],[-k, 1 + 4 * k, -k],[0,-k, 0]])
 
img_tmp2=[]
    # 作成したオペレータを基にシャープ化
for i in img_batch:
    img_tmp2.append(cv2.convertScaleAbs(cv2.filter2D(i, -1, shape_operator))) 



# In[82]: ganmma2, ganmma3


gamma2 = 1.4
gamma3 = 0.8
look_up_table3 = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table3[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma2)
ganma2=[]
for i in img_batch:
    ganma2.append(cv2.LUT(i, look_up_table3))


look_up_table4 = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table4[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma3)
ganma3=[]
for i in img_batch:
    ganma3.append(cv2.LUT(i, look_up_table4))

    
# In[82]: # shifting and rotation (change value to use many times)


rad=np.pi/90 # circumference ratio
# distance to move to x-axis
move_x = 0
# distance to move to x-axis
move_y = 96 * -0.000000005
 
matrix = [[np.cos(rad),  -1 * np.sin(rad), move_x], [np.sin(rad),   np.cos(rad), move_y]]
 
affine_matrix3 = np.float32(matrix)
afn_90=[]
for i in gaikan:
    afn_90.append(cv2.warpAffine(i, affine_matrix3, size, flags=cv2.INTER_LINEAR))


# In[83]: connection of all parts


seen5000 = np.concatenate((img_batch, low_cont_img, high_cont_img,ganma, ganma1,
                           blur_img, img_tmp, img_tmp2,ganma2, ganma3, afn_90))
seen5000.shape


# In[84]: flip right and left which its images bacome twice


flip_img=[]
for i in seen5000:
    flip_img.append(cv2.flip(i, 1))
seen10000 = np.r_[seen5000, flip_img]
plt.imshow(seen10000[1611], 'gray')#32×32
plt.show()
seen10000.shape


# In[86]: save to pickle file


with open('ロビーtrain128.pickle', mode='wb') as f:
    pickle.dump(seen10000, f)
