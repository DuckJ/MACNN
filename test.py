import numpy as np
import tensorflow as tf 


# a = np.array([[3,1,5],[3,4,2],[1,2,3],[3,5,2]])
# a = a.reshape((-1,))
# print a.shape[0]

# b = np.zeros([a.shape[0],2],dtype=int)
# print b.shape

# for index,top in enumerate(a):
# 	# print index, top
# 	b[index,:]= top%28,top/28

# print b.shape
# print b
# print a 

# b = np.argmax(a,axis=2)

# print b 

# x = b*1
# y = b*2
# aa = np.array([x,y])
# print aa.shape
# aa = np.transpose(aa,[1,2,0])
# print aa
# print aa.shape
import matplotlib.pyplot as plt

# features = np.load(open('features_train.npy','rb'))
# print features.shape 
# features = features.reshape((-1,))
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# data = np.zeros([features.shape[0],2],dtype=int)
#
# for index,top in enumerate(features):
# 	# print index, top
# 	data[index,:]= top%28,top/28

# print 'data shape', data.shape
# np.savetxt('zuobiao',data)
# data = np.loadtxt('zuobiao')
# print data.shape
# km = KMeans(n_clusters=4)
# s = km.fit(data)
# cents = km.cluster_centers_
# print cents
# # # sse = km.sse
# # print s
# # print km.cluster_centers_
# # print km.labels_
# labels = km.labels_

# colors = ['b','g','r','#000000']
# n_clusters = 4
# for i in range(n_clusters):
# 	index = np.nonzero(labels==i)[0]
# 	x0 = data[index,0]
# 	x1 = data[index,1]
# 	for j in range(1000):
# 	    plt.text(x0[j],x1[j],i,color=colors[i],fontdict={'weight': 'bold', 'size': 9})
# 	plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],linewidths=12)


# plt.axis([0,30,0,30])
# # plt.show()
# result = km.predict([[15,8],[19,20],[12,7],[19,8]])
# # #  0 1 2 3
# print '***********'
# print result
# result = km.predict([[19,20],[15,8],[1,2],[12,7]])
# print result

# joblib.dump(km, 'km.pkl')

# km = joblib.load('km.pkl')
# print km.cluster_centers_
# result = km.predict([[15,8],[19,20],[12,7],[19,8]])
# print result
# result = km.predict(data)
# # np.savetxt('channel_label',result)
# result = np.loadtxt('channel_label')
# print result.shape
# result = result.reshape((-1,512))
# print result.shape
# np.savetxt('channel_label2',result)
# result = np.loadtxt('channel_label2')
# print result
# print('****************')
# result[result!=3] = 5
# result[result==3] = 1
# result[result ==5] = 0
# print result
# np.savetxt('p3_channel',result)
# result = np.loadtxt('p3_channel')
# a = result[0:10]
# print a
# print a.shape
def generator(filename,batch_size):
	label = np.loadtxt('p0_channel')
	with open(filename, 'r') as f:
		lines = f.readlines()
		indexs = list(range(len(lines)))
		while True:
			np.random.shuffle(indexs)
			label = label[indexs]
		    # for i in range(0,100,batchsize):
			for i in range(0,len(indexs), batch_size):
				batch_x = lines[i:(i+batch_size)]
				batch_y = label[i:(i+batch_size)]
				result = np.empty((batch_size,448,448,3))
				for t,x in enumerate(batch_x):
					addr = x[:3]+'/'+x[4:]
					addr = addr.rstrip('\r\n')
					addr = addr.split(' ')[0]
					path = train_data_dir + '/'+ addr
					img = pil_image.open(path)
					img = img.convert('RGB')
					img = img.resize((448,448), pil_image.BILINEAR)
					x = np.asarray(img, dtype=K.floatx())
					result[t,:,:,:] = x
				result = result/255


				yield result,batch_y


filename = '/home/zhangjin/bird_data/train_list.txt'  
# label = np.loadtxt('p0_channel')
# label = label[:5]
# print label.shape[0]
# indexs = list(range(label.shape[0]))
# np.random.shuffle(indexs)
# # print indexs
# label = label[indexs]
# print label
import random

with open(filename, 'r') as f:
  lines = f.readlines()
  lines = lines[:5]
  print type(lines)
  print lines
  label = np.loadtxt('p0_channel')
  label = label[:5]
  print type(label)
  print label
  # cc = list(zip(lines, label))
  # random.shuffle(cc)
  # lines[:], label[:] = zip(*cc)
  # print(lines, label)
  # print lines.shape
  # print label.shape
  indexs = list(range(5))
  np.random.shuffle(indexs)
  label = label[indexs]
  # lines = lines[indexs]
  print('**************')
  print label
  print indexs
  new = []
  # for i in lines:
  # 	print i
  for i in indexs:
  	# print lines[i]
  	new.append(lines[i])
  	# print j,i
  print type(new)