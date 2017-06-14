###fit the regression 
import pandas as pd
from pandas.stats.api import ols
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
import math
from matplotlib import pyplot as plt

filepath = "/Users/eshwarchandrasekharan/Desktop/repo/predict-pop/code/"

##given performance in the previous page, where should I post next?
import pandas as pd
from pandas.stats.api import ols
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
import math

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

from sklearn import linear_model
# clf = linear_model.LinearRegression()
# print("LINEAR REGRESSION!")

#from sklearn.tree import DecisionTreeRegressor
#clf = DecisionTreeRegressor(max_depth=10)
#print("TREE REGRESSION!")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#clf =  AdaBoostClassifier()
clf = AdaBoostClassifier(
			 DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=20)
print("ADABOOST!")

log_scaling = 0
# log_scaling = 1

if log_scaling == 1:
    print("LOG SCALED!")
else:
    print("RAW COUNTS!")
train_df = pd.read_csv(filepath + 'jan_to_jun_2017_videos_type_2_16')
train_df = train_df.fillna(0)

# video_df.shape, video_df['external_id'].unique().shape
train_df = train_df.sort_values('firsthour_stats_date', ascending = False).drop_duplicates(subset=['external_id'], keep = 'last')

post_type = ['post_type_id']
one_hour_features = [
                'consumptions_by_type__link_clicks',
                'stories_by_action_type__share',
                'video_views_10s_organic',
                'video_complete_views_30s_organic',
                'video_complete_views_organic',
               ]
one_hour_features = ['firsthour_' + var for var in one_hour_features]

# train_features = one_hour_features
train_features = one_hour_features + post_type

if log_scaling == 1:
    for feats in (one_hour_features):
        train_df[feats] = np.log(train_df[feats] + 1)

# lm = linear_model.LinearRegression(fit_intercept=True, normalize=True)
cv = 10

# train_df['10s_bucket'] = np.log(train_df['twodays_video_views_10s_organic'] + 1).astype(int)
# train_df['30s_bucket'] = np.log(train_df['twodays_video_complete_views_30s_organic'] + 1).astype(int)
# train_df['complete_views_bucket'] = np.log(train_df['twodays_video_complete_views_organic'] + 1).astype(int)

print("No. of data-points = ", len(train_df))

if log_scaling == 1:
    y_share = np.log(1+train_df['twodays_stories_by_action_type__share']).astype(int)
    y_clicks = np.log(1+train_df['twodays_consumptions_by_type__link_clicks']).astype(int)
else:
    y_share = train_df['twodays_stories_by_action_type__share']
    y_clicks = train_df['twodays_consumptions_by_type__link_clicks']

# y_10s = train_df['twodays_video_views_10s_organic']
# y_30s = train_df['twodays_video_complete_views_30s_organic']
# y_complete = train_df['twodays_video_complete_views_organic']
###
# y_10s = train_df['10s_bucket']
# y_30s = train_df['30s_bucket']
# y_complete = train_df['complete_views_bucket']

###generate DFs for analysis - X and Y
X = train_df[train_features]

from sklearn.model_selection import KFold

fold = 0
cv_folds = 10
kf = KFold(n_splits = cv_folds, shuffle = True)

accuracy_10s = []
error_10s = []
accuracy_30s = []
error_30s = []
accuracy_complete = []
error_complete = []

for train_index, test_index in kf.split(X):
    print("Fold = ", fold)
#     print("Shares: Fold = ", fold)
    y = y_share
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
        
    accuracy = metrics.r2_score(y_test, y_pred)
#     print("Cross-Predicted Accuracy (R2):", accuracy)
    accuracy_10s.append(accuracy)
    from sklearn.metrics import mean_absolute_error
    # print("Mean Absolute Error: ", mean_absolute_error(y, predictions))
    error_percent = mean_absolute_error(y_test, y_pred)/y.mean()
#     print("Mean values (share): ", y_test.mean(), " | percent error: ",  error_percent)
    error_10s.append(error_percent)
    
#     print("Clicks: Fold = ", fold)
    y = y_clicks
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    
    accuracy = metrics.r2_score(y_test, y_pred)
#     print("Cross-Predicted Accuracy (R2):", accuracy)
    accuracy_30s.append(accuracy)
    from sklearn.metrics import mean_absolute_error
    # print("Mean Absolute Error: ", mean_absolute_error(y, predictions))
    error_percent = mean_absolute_error(y_test, y_pred)/y.mean()
#     print("Mean values (share): ", y_test.mean(), " | percent error: ",  error_percent)
    error_30s.append(error_percent)
    
    fold += 1
    
print("Share performance: Accuracy = ", np.mean(accuracy_10s), " ; Error (/100) = ", np.mean(error_10s))
print("Clicks performance: Accuracy = ", np.mean(accuracy_30s), " ; Error (/100) = ", np.mean(error_30s))


