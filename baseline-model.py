import numpy as np
import pandas as pd
import data_clean_utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer,MissingIndicator
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder,MinMaxScaler,PowerTransformer,OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn import set_config
set_config(transform_output="pandas")

df=pd.read_csv('F:\\new_downloads\\swiggy.csv')

data_clean_utils.perform_data_cleaning(df)

df=pd.read_csv('swiggy_cleaned.csv')
df.head()
print(df.columns)

columns_to_drop=["rider_id",'restaurant_latitude','restaurant_longitude','delivery_latitude','delivery_longitude','order_date','order_time_hour','order_day']

df.drop(columns=columns_to_drop,inplace=True)
print(df)
print(df.isna().sum())
print(df.duplicated().sum())
import missingno as msno
print(msno.matrix(df))

missing_col=(
  df.isna().any(axis=0).loc[lambda x:x].index
)
print(missing_col)

temp_df=df.copy().dropna()
X=temp_df.drop(columns='time_taken')
y=temp_df['time_taken']

print(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)

print(y_train)

print(X_train.isna().sum())
print(X_train.columns)

# do basic preprocessing

num_cols = ["age","ratings","pickup_time_minutes","distance"]

nominal_cat_cols = ['weather','type_of_order',
                    'type_of_vehicle',"festival",
                    "city_type","city_name","order_month",
                    "order_day_of_week",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]
print(len(num_cols+nominal_cat_cols+ordinal_cat_cols))

for col in ordinal_cat_cols:
  print(col,X_train[col].unique())

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]
# build a preprocessor

preprocessor = ColumnTransformer(transformers=[
    ("scale", MinMaxScaler(), num_cols),
    ("nominal_encode", OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False), nominal_cat_cols),
    ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,distance_type_order]), ordinal_cat_cols)
],remainder="passthrough",n_jobs=-1,force_int_remainder_cols=False,verbose_feature_names_out=False)

preprocessor.set_output(transform="pandas")

X_train_trans=preprocessor.fit_transform(X_train)
X_test_trans=preprocessor.transform(X_test)
print(X_train_trans)

print(y_train.values)
print(y_train.values.reshape(-1,1))
pt=PowerTransformer()
y_train_pt=pt.fit_transform(y_train.values.reshape(-1,1))
y_test_pt=pt.transform(y_test.values.reshape(-1,1))
print(pt.lambdas_)
print(y_train_pt)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_trans,y_train_pt)
# get the predictions
y_pred_train = lr.predict(X_train_trans)
y_pred_test = lr.predict(X_test_trans)

# get the actual predictions values

y_pred_train_org = pt.inverse_transform(y_pred_train.reshape(-1,1))
y_pred_test_org = pt.inverse_transform(y_pred_test.reshape(-1,1))

from sklearn.metrics import mean_absolute_error, r2_score

print(f"The train error is {mean_absolute_error(y_train,y_pred_train_org):.2f} minutes")
print(f"The test error is {mean_absolute_error(y_test,y_pred_test_org):.2f} minutes")

print(f"The train r2 score is {r2_score(y_train,y_pred_train_org):.2f}")
print(f"The test r2 score is {r2_score(y_test,y_pred_test_org):.2f}")

temp_df = df.copy()
# split into X and y

X = temp_df.drop(columns='time_taken')
y = temp_df['time_taken']

print(X)

# train test split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("The size of train data is",X_train.shape)
print("The shape of test data is",X_test.shape)

print(X_train.isna().sum())

# transform target column

pt = PowerTransformer()

y_train_pt = pt.fit_transform(y_train.values.reshape(-1,1))
y_test_pt = pt.transform(y_test.values.reshape(-1,1))
print(y_test_pt)

print(missing_col)

# percentage of rows in data having missing values

print(
    X_train
    .isna()
    .any(axis=1)
    .mean()
    .round(2) * 100
)

X_train['age'].describe()
# missing values in the column

print(X_train['age'].isna().sum())

# median value

age_median = X_train['age'].median()
print(age_median)

sns.kdeplot(X_train['age'],label='original')
sns.kdeplot(X_train['age'].fillna(age_median),label="imputed")
plt.legend()

# missing values

X_train['ratings'].isna().sum()
# statistical summary

X_train['ratings'].describe()

# avg rating

ratings_mean = X_train['ratings'].mean()

# fill and plot kdeplot

sns.kdeplot(X_train['ratings'],label="original")
sns.kdeplot(X_train['ratings'].fillna(ratings_mean),label="imputed")
plt.legend()

# value counts

X_train['weather'].value_counts()
# missing values in the column

X_train['weather'].isna().sum()
# countplot

sns.countplot(X_train['weather'])

# capture the missingness

missing_weather = MissingIndicator()
missing_weather.set_output(transform="pandas")

pd.concat([X_train['weather'],missing_weather.fit_transform(X_train[['weather']])],axis=1).sample(50)
# print(X_train['missingindicator_weather'])
# value counts

X_train['traffic'].value_counts()
# Missing values in column

X_train['traffic'].isna().sum()
# countplot

sns.countplot(X_train['traffic'])
print(missing_col)

# value counts

X_train['multiple_deliveries'].value_counts()

# countplot

sns.countplot(X_train['multiple_deliveries'].apply(str))

# number of missing values

X_train['multiple_deliveries'].isna().sum()
# mode value

multiple_deliveries_mode = X_train['multiple_deliveries'].mode()[0]

# fill na values with mode

sns.countplot(X_train['multiple_deliveries'].fillna(multiple_deliveries_mode).apply(str))

# value counts

X_train['festival'].value_counts()
# countplot

sns.countplot(X_train['festival'])

# missing values in column

X_train['festival'].isna().sum()
# mode value

festival_mode = X_train['festival'].mode()[0]
# fill with mode

sns.countplot(X_train['festival'].fillna(festival_mode))

# value counts

X_train['city_type'].value_counts()

# number of missing values

X_train['city_type'].isna().sum()

# countplot

sns.countplot(X_train['city_type'])
# mode value

city_type_mode = X_train['city_type'].mode()[0]

# fill with mode

sns.countplot(X_train['city_type'].fillna(city_type_mode))

print(missing_col)

# statistical summary

X_train['pickup_time_minutes'].describe()

# missing values in the column

X_train['pickup_time_minutes'].isna().sum()

# median value

pickup_time_minutes_median = X_train['pickup_time_minutes'].median()
# histplot

sns.histplot(X_train['pickup_time_minutes'],kde=True,label='original')
sns.histplot(X_train['pickup_time_minutes'].fillna(pickup_time_minutes_median),kde=True,label='imputed')
plt.legend()

# value counts

X_train['order_time_of_day'].value_counts()

# missing values

X_train['order_time_of_day'].isna().sum()

# countplot

sns.countplot(X_train['order_time_of_day'])

# rows where the data is missing

X_train[X_train['order_time_of_day'].isna()]

# statistical summary

X_train['distance'].describe()

# number of missing values

X_train['distance'].isna().sum()

# avg distance

distance_mean = X_train['distance'].mean()

# kdeplot

sns.kdeplot(X_train['distance'],label='original')
sns.kdeplot(X_train['distance'].fillna(distance_mean),label='imputed')
plt.legend()

print(missing_col)

# value counts

X_train['distance_type'].value_counts()

# missing values

X_train['distance_type'].isna().sum()

# countplot

sns.countplot(X_train['distance_type'])

print(nominal_cat_cols)

print(X_train.isna().sum())

# features to fill values with mode

features_to_fill_mode = ['multiple_deliveries','festival','city_type']
features_to_fill_missing = [col for col in nominal_cat_cols if col not in features_to_fill_mode]

print(features_to_fill_missing)

# simple imputer to fill categorical vars with mode

simple_imputer = ColumnTransformer(transformers=[
    ("mode_imputer",SimpleImputer(strategy="most_frequent"),features_to_fill_mode),
    ("missing_imputer",SimpleImputer(strategy="constant",fill_value="missing"),features_to_fill_missing)
],remainder="passthrough",n_jobs=-1,force_int_remainder_cols=False,verbose_feature_names_out=False)

print(simple_imputer)

simple_imputer.fit_transform(X_train)

simple_imputer.fit_transform(X_train).isna().sum()

# knn imputer

knn_imputer = KNNImputer(n_neighbors=5)

# do basic preprocessing

num_cols = ["age","ratings","pickup_time_minutes","distance"]

nominal_cat_cols = ['weather','type_of_order',
                    'type_of_vehicle',"festival",
                    "city_type","city_name","order_month",
                    "order_day_of_week",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

# generate order for ordinal encoding

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]

# unique categories the ordinal columns

for col in ordinal_cat_cols:
    print(col,X_train[col].unique())

    # build a preprocessor

preprocessor = ColumnTransformer(transformers=[
    ("scale", MinMaxScaler(), num_cols),
    ("nominal_encode", OneHotEncoder(drop="first",handle_unknown="ignore",
                                     sparse_output=False), nominal_cat_cols),
    ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,distance_type_order],
                                      encoded_missing_value=-999,
                                      handle_unknown="use_encoded_value",
                                      unknown_value=-1), ordinal_cat_cols)
],remainder="passthrough",n_jobs=-1,force_int_remainder_cols=False,verbose_feature_names_out=False)


print(preprocessor)

preprocessor.fit_transform(X_train)
# print(preprocessor.fit_transform(X_train).isna().sum().loc[lambda ser:ser.ge(0)])

print(preprocessor.fit_transform(X_train).isna().sum().loc[lambda ser : ser.ge(1)])

# build the pipeline

processing_pipeline = Pipeline(steps=[
                                ("simple_imputer",simple_imputer),
                                ("preprocess",preprocessor),
                                ("knn_imputer",knn_imputer)
                            ])

print(processing_pipeline)

# fit and transform the pipeline on X_train

print(processing_pipeline.fit_transform(X_train))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

model_pipe = Pipeline(steps=[
                                ("preprocessing",processing_pipeline),
                                ("model",lr)
                            ])

print(model_pipe)

# fit the pipeline on data

model_pipe.fit(X_train,y_train_pt)
# get the predictions
y_pred_train = model_pipe.predict(X_train)
y_pred_test = model_pipe.predict(X_test)

# get the actual predictions values

y_pred_train_org = pt.inverse_transform(y_pred_train.reshape(-1,1))
y_pred_test_org = pt.inverse_transform(y_pred_test.reshape(-1,1))

from sklearn.metrics import mean_absolute_error, r2_score

print(f"The train error is {mean_absolute_error(y_train,y_pred_train_org):.2f} minutes")
print(f"The test error is {mean_absolute_error(y_test,y_pred_test_org):.2f} minutes")

print(f"The train r2 score is {r2_score(y_train,y_pred_train_org):.2f}")
print(f"The test r2 score is {r2_score(y_test,y_pred_test_org):.2f}")

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42,n_jobs=-1)

model_pipe = Pipeline(steps=[
                                ("preprocessing",processing_pipeline),
                                ("model",rf)
                            ])

print(model_pipe)

# fit the pipeline on data

model_pipe.fit(X_train,y_train_pt.values.ravel())

# get the predictions
y_pred_train = model_pipe.predict(X_train)
y_pred_test = model_pipe.predict(X_test)

# get the actual predictions values

y_pred_train_org = pt.inverse_transform(y_pred_train.reshape(-1,1))
y_pred_test_org = pt.inverse_transform(y_pred_test.reshape(-1,1))

from sklearn.metrics import mean_absolute_error, r2_score

print(f"The train error is {mean_absolute_error(y_train,y_pred_train_org):.2f} minutes")
print(f"The test error is {mean_absolute_error(y_test,y_pred_test_org):.2f} minutes")

print(f"The train r2 score is {r2_score(y_train,y_pred_train_org):.2f}")
print(f"The test r2 score is {r2_score(y_test,y_pred_test_org):.2f}")