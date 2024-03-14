# to learn k we are using minimum error approach at which k error is minimum
# below is the algotrithm for KNN Classifier
# K Nearest Neighbors (KNN) Classifier
def KNN_Classifier(X_train_split,X_test,y_train_split,k):
    # this will store distance of k nearest point
    y_test=[]
    for j in X_test:
         dist=[]
         label_index=[] # it will return index value corresponding to k nearest neighbour
         # intitialising the initial distance to infinity
         for i in range(0,k):
           dist.append(float('inf'))
           label_index.append(-1)
         index=0
         for i in X_train_split:
           d=((i[0]-j[0])**2+(i[1]-j[1])**2)**0.5
           # updating k nearest point distances
           for l in range(0,k):
             if(dist[l]>d):
               dist[l]=d
               label_index[l]=index
               break
           index+=1
         y=0
         for index in label_index:
            y+=y_train_split[index]
         y=y/k
         if(y>=0.5):
            y_test.append(1)
         else:
            y_test.append(0)
    return np.array(y_test)
# this is MSE function
def error_loss(y_test_split,y_pred):
  min=0
  for i in range(0,len(y_test_split)):
    min+=(y_test_split[i]-y_pred[i])**2
  min_loss=min/len(y_test_split)
  return min_loss

#iterating for finding k for minimum error
error=float('inf')
k=1
# below algorithmn will train k for KNN classifier
for i in range(1,160):
  y_pred=KNN_Classifier(X_train_split,X_test,y_train_split,i)
  # print(y_pred)
  e=error_loss(y_test_split,y_pred)
  # print(e)
  if(e<=error):
    error=e
    k=i
  else:
    # print("error is too high")
    continue
  #results for the KNN classifier
print('Below is results on KNN classifier for gaussians distribution')
print(f'The value of k for minimum loss is {k}')
print(f'The value of Mean Square Error for k {k} is {error}')
