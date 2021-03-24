# -*- coding: utf-8 -*-

import warnings 
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt  
from skimage.io import imread, imshow 
from skimage.transform import resize 
from skimage.filters import sobel
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
import gc
from sklearn.utils import resample
import seaborn as sns
from skimage.color import rgb2hsv
from skimage.transform import rotate
from skimage.color import hsv2rgb
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble  import RandomForestClassifier
from skmultilearn.adapt import MLkNN
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#data = np.load('/content/gdrive/MyDrive/Intro_ML/final_project_1/data_virtualTraveling.npz')
data = np.load(r"D:\Downloads\data_virtualTraveling.npz")
X = data['X']
y = data['Y']

#inspecting how many labels and the combinations of labels are present, messy approach
"""##informations/eda##"""

overall_snow = 0
overall_mountains = 0
overall_forest = 0
overall_stars = 0
overall_desert = 0

only_snow = 0
only_mountains = 0
only_forest = 0
only_stars = 0
only_desert = 0

snow_mountains = 0
snow_forest = 0
snow_stars = 0
snow_desert = 0

mountains_forest = 0
mountains_stars = 0
mountains_desert = 0

forest_stars = 0
forest_desert = 0

stars_desert = 0

snow_mountains_forest = 0
snow_mountains_stars = 0
snow_mountains_desert = 0
snow_forest_stars = 0
snow_forest_desert = 0
snow_stars_desert = 0

mountains_forest_stars = 0
mountains_forest_desert = 0
mountains_stars_desert = 0

forest_stars_desert = 0

snow_mountains_forest_stars_desert = 0


where_only_snow = []
where_only_mountains = []
where_only_forest = []
where_only_stars  = []
where_only_desert = []

where_snow_mountains = []
where_snow_forest = []

where_mountains_forest = []
where_mountains_stars = []
where_mountains_desert = []

where_forest_desert = []

where_snow_mountains_forest = []

where_mountains_stars_desert = []

for t in range(0, len(y)):
  snow = 0
  mountains = 0
  forest = 0
  stars = 0
  desert = 0
  i = 0
  if y[t, i] == 1:
    snow +=1
  i += 1
  if y[t, i ] == 1:
    mountains += 1
  i += 1
  if y[t, i ] == 1:
    forest += 1
  i += 1
  if y[t, i ] == 1:
    stars += 1
  i += 1
  if y[t, i ] == 1:
    desert += 1

  if snow == 1:
    overall_snow += 1
  if mountains == 1:
    overall_mountains += 1
  if forest == 1:
    overall_forest += 1
  if stars == 1:
    overall_stars += 1
  if desert == 1:
    overall_desert += 1

################################################################################  
  if snow == 1 and (mountains+forest+stars+desert) == 0:
    only_snow += 1
    where_only_snow.append(t)
  if mountains == 1 and (snow+forest+stars+desert) == 0:
    only_mountains += 1
    where_only_mountains.append(t)
  if forest == 1 and (mountains+snow+stars+desert) == 0:
    only_forest += 1
    where_only_forest.append(t)
  if stars == 1 and (mountains+forest+snow+desert) == 0:
    only_stars += 1
    where_only_stars.append(t)
  if desert == 1 and (mountains+forest+stars+snow) == 0:
    only_desert += 1
    where_only_desert.append(t)
################################################################################
  if (snow == 1 and mountains == 1) and (forest+desert+stars) == 0:
     snow_mountains += 1
     where_snow_mountains.append(t)
  if (snow == 1 and forest == 1) and (mountains+desert+stars) == 0:
     snow_forest += 1
     where_snow_forest.append(t)
  if (snow == 1 and stars== 1) and (forest+desert+mountains) == 0:
     snow_stars += 1

  if (snow == 1 and desert == 1) and (forest+mountains+stars) == 0:
     snow_desert += 1

################################################################################
  if (mountains == 1 and forest == 1) and (snow+desert+stars) == 0:
     mountains_forest += 1
     where_mountains_forest.append(t)

  if (mountains == 1 and stars == 1) and (snow+desert+forest) == 0:
     mountains_stars += 1
     where_mountains_stars.append(t)

  if (mountains == 1 and desert== 1) and (forest+snow+stars) == 0:
     mountains_desert += 1
     where_mountains_desert.append(t)
################################################################################
  if (forest == 1 and stars == 1) and (snow+mountains+desert) == 0:
     forest_stars += 1

  if (forest == 1 and desert == 1) and (snow+mountains+stars) == 0:
     forest_desert += 1
     where_forest_desert.append(t)
################################################################################
  if (stars == 1 and desert== 1) and (snow+mountains+forest) == 0:
     stars_desert += 1
################################################################################
  if (snow == 1 and mountains == 1 and forest ==1 ) and (stars+desert) == 0:
      snow_mountains_forest += 1
      where_snow_mountains_forest = t
  if (snow == 1 and mountains == 1 and stars ==1 ) and (forest+desert) == 0:
      snow_mountains_stars += 1
  if (snow == 1 and mountains == 1 and desert ==1 ) and (forest+stars) == 0:
      snow_mountains_desert += 1
  if (snow == 1 and forest == 1 and stars ==1 ) and (mountains+desert) == 0:
      snow_forest_stars += 1
  if (snow == 1 and forest == 1 and desert ==1 ) and (mountains+stars) == 0:
      snow_forest_desert += 1     
  if (snow == 1 and stars == 1 and desert ==1 ) and (mountains+forest) == 0:
      snow_stars_desert += 1 
  ################################################################################
  if (mountains == 1 and forest == 1 and stars ==1 ) and (snow+desert) == 0:
      mountains_forest_stars += 1 
  if (mountains == 1 and forest == 1 and desert ==1 ) and (snow+stars) == 0:
      mountains_forest_desert += 1 
  if (mountains == 1 and stars == 1 and desert ==1 ) and (snow+forest) == 0:
      mountains_stars_desert += 1 
      where_mountains_stars_desert.append(t)
  ################################################################################
  if (forest == 1 and stars == 1 and desert ==1 ) and (snow+mountains) == 0:
      forest_stars_desert += 1 
  ################################################################################
  if (forest == 1 and stars == 1 and desert == 1 and snow == 1 and mountains == 1):
    snow_mountains_forest_stars_desert += 1
    

print(f'overall_snow: {overall_snow}')
print(f'overall_mountains: {overall_mountains}')
print(f'overall_forest: {overall_forest}')
print(f'overall_stars: {overall_stars}')
print(f'overall_desert: {overall_desert}')
print(f'Overall_sum: {overall_snow + overall_mountains + overall_forest + overall_stars + overall_desert}')

print(f'only_snow: {only_snow}')
print(f'where_only_snow: {where_only_snow}')
print(f'only_mountains: {only_mountains}')
print(f'where_only_mountains: {where_only_mountains}')
print(f'only_forest: {only_forest}')
print(f'where_only_forest: {where_only_forest}')
print(f'only_stars: {only_stars}')
print(f'where_only_stars: {where_only_stars}')
print(f'only_desert: {only_desert}')
print(f'where_only_desert: {where_only_desert}')

print(f'snow and mountains: {snow_mountains}')
print(f'where_snow_mountains: {where_snow_mountains}')
print(f'snow and forest: {snow_forest}')
print(f'where_snow_forest: {where_snow_forest}')
print(f'snow and stars: {snow_stars}')
print(f'snow and desert: {snow_desert}')

print(f'mountains and forest: {mountains_forest}')
print(f'where_mountains_forest: {where_mountains_forest}')
print(f'mountains and stars: {mountains_stars}')
print(f'where_mountains_stars: {where_mountains_stars}')
print(f'mountains and desert: {mountains_desert}')
print(f'where_mountains_desert: {where_mountains_desert}')

print(f'forest and stars: {forest_stars}')
print(f'forest and desert: {forest_desert}')
print(f'where_forest_desert: {where_forest_desert}')

print(f'stars and desert: {stars_desert}')

print(f'snow, mountains and forest: {snow_mountains_forest}')
print(f'snow, mountains and stars: {snow_mountains_stars}')
print(f'snow, mountains and desert: {snow_mountains_desert}')
print(f'snow, forest and stars: {snow_forest_desert}')
print(f'snow, forest and desert: {snow_forest_desert}')
print(f'snow, stars and desert: {snow_stars_desert}')

print(f'mountains, forest and stars: {mountains_forest_stars}')
print(f'mountains, forest and desert: {mountains_forest_desert}')
print(f'mountains, stars and desert: {mountains_stars_desert}')
print(f'where_mountains_stars_desert: {where_mountains_stars_desert}')

print(f'forest, stars and desert: {forest_stars_desert}')

print(f'snow, mountains, forest, stars and desert: {snow_mountains_forest_stars_desert}')

# plot of labels occurence
#https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff
df3 = pd.DataFrame.from_records(y)
df3.rename(classes,axis=1, inplace=True)
df = pd.merge(df1, df3, left_index=True, right_index=True)

categories = list(df.iloc[:,1:].columns.values)
categories

categories = list(df.iloc[:,1:].columns.values)
sns.set(font_scale = 2)
plt.figure(figsize=(15,8))
ax= sns.barplot(categories, df.iloc[:,1:].sum().values)
plt.title("Occurence in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Type ', fontsize=18)
#adding the text labels
rects = ax.patches
labels = df.iloc[:,1:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.show()

rowSums = df.iloc[:,1:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[1:]
sns.set(font_scale = 2)
plt.figure(figsize=(15,8))
ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Images having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)
#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()

"""##models##"""

gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=20)

#make big function for feature extraction
def feature_extraction(dataset):
  SIZE = 128
  image_dataset = pd.DataFrame()
  for image in range(dataset.shape[0]):
    df = pd.DataFrame()
    input_img = dataset[image, :, :, :]
    img = input_img
    #print(img.shape)
    img = resize(img, (SIZE, SIZE))
    pixel_values = img.reshape(-1)
    df['Original_Pixel_Values'] = pixel_values
    #print(df.shape)
    hsv_img = rgb2hsv(img)
    hue_img = hsv_img[:, :, 0]
    saturation_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]

    # median filter on hsv channels
    hue_median = nd.median_filter(hue_img, size=3)
    saturation_median = nd.median_filter(saturation_img, size=3)
    value_median = nd.median_filter(value_img, size=3)
    filtered_img1 = np.dstack((hue_median, saturation_median, value_median))

    median_filtered_img_rgb = hsv2rgb(filtered_img1)

    #sobel on median filtered hsv channels   
    sobel_hue = sobel(hue_median)
    sobel_saturation = sobel(saturation_median)
    sobel_value = sobel(value_median)
    sobel_filtered_img_hsv = np.dstack((sobel_hue, sobel_saturation, sobel_value))
    sobel_filtered_img_hsv1 = sobel_filtered_img_hsv.reshape(-1)
    df['sobel_filtered'] = sobel_filtered_img_hsv1

    #https://www.youtube.com/watch?v=uWTzkUD3V9g
    num = 1
    kernels = []
    for theta in range(2):
      theta = theta / 4. * np.pi
      for sigma in (1, 3):
        lamda = np.pi/4
        gamma = 0.5
        gabor_label = 'Gabor' + str(num)
        ksize = 9
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
        fimg = cv2.filter2D(median_filtered_img_rgb, cv2.CV_8UC3, kernel)
        filtered_img  = fimg.reshape(-1)
        df[gabor_label] = filtered_img
        print(gabor_label, ': theta=', theta, ': sigma:', sigma, ': lamda=', lamda, ': gamma=', gamma)
        num += 1  

    image_dataset = image_dataset.append(df)

  return image_dataset

image_features = feature_extraction(X_train)

gc.collect()

test_features = feature_extraction(X_test)

gc.collect()

n_features = image_features.shape[1] # just take the features
image_features1 = np.expand_dims(image_features, axis = 0) 
X_for_RF = np.reshape(image_features1, (X_train.shape[0], -1))

test_features1 = np.expand_dims(test_features, axis = 0)
test_for_RF =  np.reshape(test_features1, (X_test.shape[0], -1))

gc.collect()

"""random forest classifier"""

scoring = 'f1_weighted'
param_grid = {'bootstrap': [True,False],
              'n_estimators' : [50, 100, 200, 500],
              'max_depth': [3, 5, 8],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 8]}

rf_model = RandomForestClassifier(class_weight='balanced', random_state = 42)
rf_rsv = RandomizedSearchCV(rf_model, param_grid, scoring=scoring, cv = 3, random_state= 42)
rf_rsv.fit(X_for_RF, y_train)

rf_rsv.best_params_

gc.collect()

scores_forest = cross_validate(rf_rsv.best_estimator_, X_for_RF, y_train, scoring=scoring, cv=3)

scores_forest

print(f"RandomForest Classification Train Set \nf1_weighted: {scores_forest['test_score'].mean().round(3)}")

gc.collect()

best_rf = rf_rsv.best_estimator_
test_prediction_rf = best_rf.predict(test_for_RF)

f1_weighted = f1_score(y_test, test_prediction_rf, average='weighted')
print(f"RandomForest Classification Test Set \nf1_weighted {f1_weighted.round(3)}")

"""knn"""

gc.collect()

parameters = {'k': range(1,4), 's': [0.5, 0.7, 1.0]}
knn_rsv = RandomizedSearchCV(MLkNN(), parameters, scoring=scoring, refit='f1_weighted')
knn_rsv.fit(X_for_RF, y_train)

knn_rsv.best_estimator_

knn_rsv.best_score_

gc.collect()

scores_knn = cross_validate(knn_rsv.best_estimator_, X_for_RF, y_train, scoring=scoring, cv=3)

print(f"MLkNN Classification Train Set \nf1_macro: {scores_knn['test_score'].mean().round(3)}")

gc.collect()

best_knn = knn_rsv.best_estimator_
test_prediction_knn = best_knn.predict(test_for_RF)

f1_weighted = f1_score(y_test, test_prediction_knn, average='weighted')
print(f"ML-kNN Test Set \nf1_samples: {f1_weighted.round(3)}")

"""Naive Bayes"""

gc.collect()

model_to_set = MultiOutputClassifier(MultinomialNB())
param_grid = {'estimator__alpha': [0.01, 0.1, 1, 2],
              'estimator__fit_prior' : [True, False]}

nb_rsv = RandomizedSearchCV(model_to_set, param_grid, scoring=scoring, cv = 3, random_state= 42)
nb_rsv.fit(X_for_RF, y_train)

nb_rsv.best_estimator_

gc.collect()

scores_nb = cross_validate(nb_rsv.best_estimator_, X_for_RF, y_train, scoring=scoring, cv=3)

print(f"MNB Classification Train Set \nf1_weighted: {scores_nb['test_score'].mean().round(3)}")

gc.collect()

best_nb = nb_rsv.best_estimator_
test_prediction_mnb = best_nb.predict(test_for_RF)

f1_weighted = f1_score(y_test, test_prediction_mnb, average='weighted')
print(f"MNB Classification Test Set \nf1_weighted: {f1_weighted.round(3)}")

"""logistic """

gc.collect()

model_to_set = MultiOutputClassifier(LogisticRegression())
param_grid = {'estimator__penalty': ['elasticnet'],
              'estimator__class_weight' : ['balanced'],
              'estimator__solver' : ['saga'],
              'estimator__C' : [10, 1.0, 0.1],
              'estimator__l1_ratio' : [0.5]}

logr_rsv = RandomizedSearchCV(model_to_set, param_grid, scoring=scoring, cv = 3, random_state= 42, verbose=10)
logr_rsv.fit(X_for_RF, y_train)

logr_rsv.best_estimator_

gc.collect()

scores_nb = cross_validate(logr_rsv.best_estimator_, X_for_RF, y_train, scoring=scoring, cv=3, verbose = 10)

print(f"Logistic Regression Classification Train Set \nf1_weighted: {scores_nb['test_score'].mean().round(3)}")

gc.collect()

best_logreg = logr_rsv.best_estimator_
test_prediction_logr = best_logreg.predict(test_for_RF)

f1_weighted = f1_score(y_test, test_prediction_logr, average='weighted')
print(f"Logistic Regression Classification Test Set \nf1_weighted: {f1_weighted.round(3)}")

"""##Confusionmatrix##
"""
# get information of labels in test set
overall_snow = 0
overall_mountains = 0
overall_forest = 0
overall_stars = 0
overall_desert = 0

only_snow = 0
only_mountains = 0
only_forest = 0
only_stars = 0
only_desert = 0

snow_mountains = 0
snow_forest = 0
snow_stars = 0
snow_desert = 0

mountains_forest = 0
mountains_stars = 0
mountains_desert = 0

forest_stars = 0
forest_desert = 0

stars_desert = 0

snow_mountains_forest = 0
snow_mountains_stars = 0
snow_mountains_desert = 0
snow_forest_stars = 0
snow_forest_desert = 0
snow_stars_desert = 0

mountains_forest_stars = 0
mountains_forest_desert = 0
mountains_stars_desert = 0

forest_stars_desert = 0

snow_mountains_forest_stars_desert = 0


where_only_snow = []
where_only_mountains = []
where_only_forest = []
where_only_stars  = []
where_only_desert = []

where_snow_mountains = []
where_snow_forest = []

where_mountains_forest = []
where_mountains_stars = []
where_mountains_desert = []

where_forest_desert = []

where_snow_mountains_forest = []

where_mountains_stars_desert = []

for t in range(0, len(y_test)):
  snow = 0
  mountains = 0
  forest = 0
  stars = 0
  desert = 0
  i = 0
  if y[t, i] == 1:
    snow +=1
  i += 1
  if y[t, i ] == 1:
    mountains += 1
  i += 1
  if y[t, i ] == 1:
    forest += 1
  i += 1
  if y[t, i ] == 1:
    stars += 1
  i += 1
  if y[t, i ] == 1:
    desert += 1

  if snow == 1:
    overall_snow += 1
  if mountains == 1:
    overall_mountains += 1
  if forest == 1:
    overall_forest += 1
  if stars == 1:
    overall_stars += 1
  if desert == 1:
    overall_desert += 1

################################################################################  
  if snow == 1 and (mountains+forest+stars+desert) == 0:
    only_snow += 1
    where_only_snow.append(t)
  if mountains == 1 and (snow+forest+stars+desert) == 0:
    only_mountains += 1
    where_only_mountains.append(t)
  if forest == 1 and (mountains+snow+stars+desert) == 0:
    only_forest += 1
    where_only_forest.append(t)
  if stars == 1 and (mountains+forest+snow+desert) == 0:
    only_stars += 1
    where_only_stars.append(t)
  if desert == 1 and (mountains+forest+stars+snow) == 0:
    only_desert += 1
    where_only_desert.append(t)
################################################################################
  if (snow == 1 and mountains == 1) and (forest+desert+stars) == 0:
     snow_mountains += 1
     where_snow_mountains.append(t)
  if (snow == 1 and forest == 1) and (mountains+desert+stars) == 0:
     snow_forest += 1
     where_snow_forest.append(t)
  if (snow == 1 and stars== 1) and (forest+desert+mountains) == 0:
     snow_stars += 1

  if (snow == 1 and desert == 1) and (forest+mountains+stars) == 0:
     snow_desert += 1

################################################################################
  if (mountains == 1 and forest == 1) and (snow+desert+stars) == 0:
     mountains_forest += 1
     where_mountains_forest.append(t)

  if (mountains == 1 and stars == 1) and (snow+desert+forest) == 0:
     mountains_stars += 1
     where_mountains_stars.append(t)

  if (mountains == 1 and desert== 1) and (forest+snow+stars) == 0:
     mountains_desert += 1
     where_mountains_desert.append(t)
################################################################################
  if (forest == 1 and stars == 1) and (snow+mountains+desert) == 0:
     forest_stars += 1

  if (forest == 1 and desert == 1) and (snow+mountains+stars) == 0:
     forest_desert += 1
     where_forest_desert.append(t)
################################################################################
  if (stars == 1 and desert== 1) and (snow+mountains+forest) == 0:
     stars_desert += 1
################################################################################
  if (snow == 1 and mountains == 1 and forest ==1 ) and (stars+desert) == 0:
      snow_mountains_forest += 1
      where_snow_mountains_forest = t
  if (snow == 1 and mountains == 1 and stars ==1 ) and (forest+desert) == 0:
      snow_mountains_stars += 1
  if (snow == 1 and mountains == 1 and desert ==1 ) and (forest+stars) == 0:
      snow_mountains_desert += 1
  if (snow == 1 and forest == 1 and stars ==1 ) and (mountains+desert) == 0:
      snow_forest_stars += 1
  if (snow == 1 and forest == 1 and desert ==1 ) and (mountains+stars) == 0:
      snow_forest_desert += 1     
  if (snow == 1 and stars == 1 and desert ==1 ) and (mountains+forest) == 0:
      snow_stars_desert += 1 
  ################################################################################
  if (mountains == 1 and forest == 1 and stars ==1 ) and (snow+desert) == 0:
      mountains_forest_stars += 1 
  if (mountains == 1 and forest == 1 and desert ==1 ) and (snow+stars) == 0:
      mountains_forest_desert += 1 
  if (mountains == 1 and stars == 1 and desert ==1 ) and (snow+forest) == 0:
      mountains_stars_desert += 1 
      where_mountains_stars_desert.append(t)
  ################################################################################
  if (forest == 1 and stars == 1 and desert ==1 ) and (snow+mountains) == 0:
      forest_stars_desert += 1 
  ################################################################################
  if (forest == 1 and stars == 1 and desert == 1 and snow == 1 and mountains == 1):
    snow_mountains_forest_stars_desert += 1
    

print(f'overall_snow: {overall_snow}')
print(f'overall_mountains: {overall_mountains}')
print(f'overall_forest: {overall_forest}')
print(f'overall_stars: {overall_stars}')
print(f'overall_desert: {overall_desert}')
print(f'Overall_sum: {overall_snow + overall_mountains + overall_forest + overall_stars + overall_desert}')

print(f'only_snow: {only_snow}')
print(f'where_only_snow: {where_only_snow}')
print(f'only_mountains: {only_mountains}')
print(f'where_only_mountains: {where_only_mountains}')
print(f'only_forest: {only_forest}')
print(f'where_only_forest: {where_only_forest}')
print(f'only_stars: {only_stars}')
print(f'where_only_stars: {where_only_stars}')
print(f'only_desert: {only_desert}')
print(f'where_only_desert: {where_only_desert}')

print(f'snow and mountains: {snow_mountains}')
print(f'where_snow_mountains: {where_snow_mountains}')
print(f'snow and forest: {snow_forest}')
print(f'where_snow_forest: {where_snow_forest}')
print(f'snow and stars: {snow_stars}')
print(f'snow and desert: {snow_desert}')

print(f'mountains and forest: {mountains_forest}')
print(f'where_mountains_forest: {where_mountains_forest}')
print(f'mountains and stars: {mountains_stars}')
print(f'where_mountains_stars: {where_mountains_stars}')
print(f'mountains and desert: {mountains_desert}')
print(f'where_mountains_desert: {where_mountains_desert}')

print(f'forest and stars: {forest_stars}')
print(f'forest and desert: {forest_desert}')
print(f'where_forest_desert: {where_forest_desert}')

print(f'stars and desert: {stars_desert}')

print(f'snow, mountains and forest: {snow_mountains_forest}')
print(f'snow, mountains and stars: {snow_mountains_stars}')
print(f'snow, mountains and desert: {snow_mountains_desert}')
print(f'snow, forest and stars: {snow_forest_desert}')
print(f'snow, forest and desert: {snow_forest_desert}')
print(f'snow, stars and desert: {snow_stars_desert}')

print(f'mountains, forest and stars: {mountains_forest_stars}')
print(f'mountains, forest and desert: {mountains_forest_desert}')
print(f'mountains, stars and desert: {mountains_stars_desert}')
print(f'where_mountains_stars_desert: {where_mountains_stars_desert}')

print(f'forest, stars and desert: {forest_stars_desert}')

print(f'snow, mountains, forest, stars and desert: {snow_mountains_forest_stars_desert}')
"""RF"""
multilabel_confusion_matrix(y_test, test_prediction_rf)


"""ML-kNN"""

multilabel_confusion_matrix(y_test, test_prediction_knn)

"""mnb"""

multilabel_confusion_matrix(y_test, test_prediction_mnb)

"""logreg"""

multilabel_confusion_matrix(y_test, test_prediction_logr)
