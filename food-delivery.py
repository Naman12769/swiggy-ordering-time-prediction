import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

df=pd.read_csv('F:\\new_downloads\\swiggy.csv')
df.head()
rows,cols=df.shape
print(rows)
print(cols)
print(df.head())
print(df.info())
print(df.describe())

print(df.isna().sum())
print(df.loc[43317,"Delivery_person_Ratings"])

print((df=="NaN ").sum().sum())
print(df.replace('NaN ',np.nan).isna().sum().sum())

print((df.loc[:,"Weatherconditions"].str.replace("conditions ","").replace("NaN",np.nan).isna().sum()))
# print(df[:,'conditions '])
missing_df = (
    df.replace("NaN ",np.nan)
    .assign(
        Weatherconditions = lambda df_ : (
            df_['Weatherconditions']
            .str.replace("conditions ","")
            .replace("NaN",np.nan)
            )
    )
)

# missing values in data

missing_df.isna().sum()
# total missing values in data

missing_df.isna().sum().sum()
# missing values matrix

print(msno.matrix(missing_df))

# bar plot for columns having missing data

print(msno.bar(missing_df))

missing_df[["Weatherconditions","Road_traffic_density"]].isna().sum()

# prove point of missingness

(
    missing_df[["Weatherconditions","Road_traffic_density"]]
    .isna().all(axis=1)
    .sum()
) /  missing_df[["Weatherconditions","Road_traffic_density"]].isna().sum()

# dendrogram of missingness

msno.dendrogram(missing_df)

# percentage of rows in the data having missing value

(missing_df.isna().any(axis=1).sum() / missing_df.shape[0]) * 100

# column names in data

df.columns.tolist()

def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"},axis=1)
    )
# change column names

df = change_column_names(df)

# check for duplicate rows

df.drop(columns=["id","rider_id"]).duplicated().sum()
# unique items in ID column

print(f"The number of unique IDs are {df['id'].nunique()}")

# unique rider ids in the column

df['rider_id'].nunique()

# rider order count in data

df['rider_id'].value_counts()

# extract city name out of rider id

print((
    df['rider_id']
    .str.split("RES")
    .str.get(0)
    .rename("City_Name")
))

df['age'].dtype

# min, mean and max values

df['age'].astype(float).describe()
# boxplot of the age column

sns.boxplot(df['age'].astype(float))

# rows of data where rider age is less than 18(minor)


minors_data = df.loc[df['age'].astype('float') < 18]

minors_data

# rows of minors

minor_index = minors_data.index.tolist()

len(minor_index)

# datatype

df['ratings'].dtype

# min, mean and max values

df['ratings'].astype(float).describe()

# boxplot

sns.boxplot(df['ratings'].astype(float))

# rows where the star rating is 6

six_star_data = df.loc[df['ratings'] == "6"]

len(six_star_data)

six_star_index = six_star_data.index.tolist()
location_columns = df.columns[4:8].tolist()

location_columns

location_subset = df.loc[:,location_columns]
location_subset

# statistical analysis

location_subset.describe()
# set the lower bound limits for the lat and long

lower_bound_lat = 6.44
lower_bound_long = 68.70

# rows of data where latitude and longitude values are below the bounds

df.loc[
    (df['restaurant_latitude'] < lower_bound_lat) |
    (df['restaurant_longitude'] < lower_bound_long) |
    (df['delivery_latitude'] < lower_bound_lat) |
    (df['delivery_longitude'] < lower_bound_long)
].sample(50)

# number of rows in data where lat long are erroneous

location_subset.loc[
    (location_subset['restaurant_latitude'] < lower_bound_lat) |
    (location_subset['restaurant_longitude'] < lower_bound_long) |
    (location_subset['delivery_latitude'] < lower_bound_lat) |
    (location_subset['delivery_longitude'] < lower_bound_long)
].shape[0]

# statistical summary of problematic rows where lat long is below the country's geographical limits

location_subset.loc[
    (location_subset['restaurant_latitude'] < lower_bound_lat) |
    (location_subset['restaurant_longitude'] < lower_bound_long) |
    (location_subset['delivery_latitude'] < lower_bound_lat) |
    (location_subset['delivery_longitude'] < lower_bound_long)
].describe()

# boxplots for all the anomalies

location_subset.loc[
    (location_subset['restaurant_latitude'] < lower_bound_lat) |
    (location_subset['restaurant_longitude'] < lower_bound_long) |
    (location_subset['delivery_latitude'] < lower_bound_lat) |
    (location_subset['delivery_longitude'] < lower_bound_long)
].plot(kind="box")

plt.xticks(rotation=45)

# taking the absolute values

(
    location_subset.abs()
    .plot(kind="box")
)

ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

# number of rows after taking absolute values

(
    location_subset.abs()
    .loc[lambda df_:
        (df_['restaurant_latitude'] < lower_bound_lat) |
        (df_['restaurant_longitude'] < lower_bound_long) |
        (df_['delivery_latitude'] < lower_bound_lat) |
        (df_['delivery_longitude'] < lower_bound_long)]
    .shape[0]

)

# lat long values less than 1
print(df)

location_subset.abs().loc[lambda df_:
                        (df_['restaurant_latitude'] < 1) |
                        (df_['restaurant_longitude'] < 1) |
                        (df_['delivery_latitude'] < 1) |
                        (df_['delivery_longitude'] < 1)]

def clean_lat_long(data: pd.DataFrame, threshold=1):
    location_columns = location_subset.columns.tolist()

    return (
        data
        .assign(**{
            col: (
                np.where(data[col] < threshold, np.nan, data[col].values)
            )
            for col in location_columns
        })
    )
clean_lat_long(df).isna().sum()

# check for missing values

df['order_date'].isna().sum()
# unique values in order date

df['order_date'].unique()
# date range

order_date = pd.to_datetime(df['order_date'],dayfirst=True)

order_date.max() - order_date.min()

# min and maximum dates

order_date.agg(["min","max"]).set_axis(["start","end"],axis=0)
# extract day, day name, month and year

def extract_datetime_features(ser):
    date_col = pd.to_datetime(ser,dayfirst=True)

    return (
        pd.DataFrame(
            {
                "day": date_col.dt.day,
                "month": date_col.dt.month,
                "year": date_col.dt.year,
                "day_of_week": date_col.dt.day_name(),
                "is_weekend": date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
            }
        ))
extract_datetime_features(df['order_date'])

# extract hour info from data

order_time_hr = pd.to_datetime(df.replace("NaN ",np.nan)['order_time'],format='mixed').dt.hour

order_time_hr

def time_of_day(ser):
    time_col = pd.to_datetime(ser,format='mixed').dt.hour

    return(
        np.select(condlist=[(ser.between(6,12,inclusive='left')),
                            (ser.between(12,17,inclusive='left')),
                            (ser.between(17,20,inclusive='left')),
                            (ser.between(20,24,inclusive='left'))],
                  choicelist=["morning","afternoon","evening","night"],
                  default="after_midnight")
    )

time_subset = df.loc[:,["order_time","order_picked_time"]]
time_subset

(
    time_subset
    .dtypes
)

time_subset.columns.tolist()

# calculate the pickup time

(
    time_subset
    .assign(**{
        col: pd.to_datetime(time_subset[col].replace("NaN ",np.nan).dropna(),format="mixed")
        for col in time_subset.columns.tolist()}
    )
    .assign(
        pickup_time = lambda x: (x['order_picked_time'] - x['order_time']).dt.seconds / 60,
        order_time_hour = lambda x: x['order_time'].dt.hour,
        order_time_of_day = lambda x: x['order_time_hour'].pipe(time_of_day)
    )
    .drop(columns=["order_time","order_picked_time"])
)
# value counts

df['weather'].value_counts()
# unique values
df['weather'].unique()
# remove conditions from values

(
    df['weather']
    .str.replace("conditions ","")
    .unique()
)
# value counts

df['traffic'].value_counts()
# unique values
df['traffic'].unique()
(
    df['traffic']
    .replace("NaN ",np.nan)
    .str.rstrip()
    .str.lower()
    .unique()
)
# unique values in column

np.sort(df['vehicle_condition'].unique())

# value counts

df['type_of_order'].value_counts()
# unique values
df['type_of_order'].unique()
(
    df['type_of_order']
    .str.rstrip()
    .str.lower()
    .unique()
)

# value counts

df['type_of_vehicle'].value_counts()

# unique values
df['type_of_vehicle'].unique()

(
    df['type_of_vehicle']
    .str.rstrip()
    .str.lower()
    .unique()
)

# datatype of multiple deliveries column

df['multiple_deliveries'].dtype

# unique values in column

df['multiple_deliveries'].unique()

# make the column as integer

(
    df['multiple_deliveries']
    .replace("NaN ",np.nan)
    .astype(float)
    .unique()
)

# unique values in column

df['festival'].unique()

(
    df['festival']
    .replace("NaN ",np.nan)
    .str.rstrip()
    .str.lower()
    .unique()
)

# unique values in city type

df['city_type'].unique()

(
    df['city_type']
    .replace("NaN ",np.nan)
    .str.rstrip()
    .str.lower()
    .unique()
)

# datatype of time taken

df['time_taken'].dtype

(
    df['time_taken']
    .str.replace("(min) ","")
    .astype(int)
)

df.columns

def data_cleaning(data: pd.DataFrame):

    return (
        data
        .drop(columns="id")
        .drop(index=minor_index)                                                # Minor riders in data dropped
        .drop(index=six_star_index)                                             # six star rated drivers dropped
        .replace("NaN ",np.nan)                                                 # missing values in the data
        .assign(
            # city column out of rider id
            city_name = lambda x: x['rider_id'].str.split("RES").str.get(0),
            # convert age to float
            age = lambda x: x['age'].astype(float),
            # convert ratings to float
            ratings = lambda x: x['ratings'].astype(float),
            # absolute values for location based columns
            restaurant_latitude = lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude = lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude = lambda x: x['delivery_latitude'].abs(),
            delivery_longitude = lambda x: x['delivery_longitude'].abs(),
            # order date to datetime and feature extraction
            order_date = lambda x: pd.to_datetime(x['order_date'],
                                                  dayfirst=True),
            order_day = lambda x: x['order_date'].dt.day,
            order_month = lambda x: x['order_date'].dt.month,
            order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend = lambda x: (x['order_date']
                                    .dt.day_name()
                                    .isin(["Saturday","Sunday"])
                                    .astype(int)),
            # time based columns
            order_time = lambda x: pd.to_datetime(x['order_time'],
                                                  format='mixed'),
            order_picked_time = lambda x: pd.to_datetime(x['order_picked_time'],
                                                         format='mixed'),
            # time taken to pick order
            pickup_time_minutes = lambda x: (
                                            (x['order_picked_time'] - x['order_time'])
                                            .dt.seconds / 60
                                            ),
            # hour in which order was placed
            order_time_hour = lambda x: x['order_time'].dt.hour,
            # time of the day when order was placed
            order_time_of_day = lambda x: (
                                x['order_time_hour'].pipe(time_of_day)),
            # categorical columns
            weather = lambda x: (
                                x['weather']
                                .str.replace("conditions ","")
                                .str.lower()
                                .replace("nan",np.NaN)),
            traffic = lambda x: x["traffic"].str.rstrip().str.lower(),
            type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),
            type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
            festival = lambda x: x['festival'].str.rstrip().str.lower(),
            city_type = lambda x: x['city_type'].str.rstrip().str.lower(),
            # multiple deliveries column
            multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),
            # target column modifications
            time_taken = lambda x: (x['time_taken']
                                    .str.replace("(min) ","")
                                    .astype(int)))
        .drop(columns=["order_time","order_picked_time"])
    )

data_cleaning(df)
location_subset.columns.tolist()

def calculate_haversine_distance(df):
    location_columns = location_subset.columns.tolist()
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )
# add more data cleaning steps

cleaned_data = (
                df.pipe(data_cleaning)
                .pipe(clean_lat_long)
                .pipe(calculate_haversine_distance)
                )

cleaned_data

# age column

cleaned_data['age'].agg(["min","max"])

# ratings column

cleaned_data['ratings'].agg(["min","max"])

# location columns

# values in categorical columns

cat_cols = cleaned_data.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    print(f"For {col} unique values are: {cleaned_data[col].unique()}",end="\n\n")
  

# bar plot of missing values

msno.bar(cleaned_data)

# matrix of missing values

msno.matrix(cleaned_data)

# correlation chart of missing values
msno.heatmap(cleaned_data)

# save the cleaned data

cleaned_data.to_csv("cleaned_data.csv",index=False)
# load the cleaned data

cleaned_data_load = pd.read_csv("cleaned_data.csv")
# data types of cleaned data

print(cleaned_data_load.dtypes)

print(cleaned_data_load)