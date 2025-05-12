import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import data_clean_utils
from scipy.stats import chi2_contingency,f_oneway,probplot,jarque_bera
import plotly.express as px

df=pd.read_csv("F:\\new_downloads\\swiggy.csv")
df.sample(30)

data_clean_utils.perform_data_cleaning(data=df)
df_final=pd.read_csv('swiggy_cleaned.csv')
print(df_final.head())
print(df_final.isna().sum())

missing_rows=(
  df_final.isnull().any(axis=1).sum()
)
print(missing_rows)
print((missing_rows/df_final.shape[0])*100)
# print(df_final.columns[[1,2]].tolist())
num_cols=df_final.columns[[1,2,3,4,5,6,16,22,25]].tolist()
cat_cols=[col for col in df_final.columns.tolist() if col not in num_cols]
print(f'there are {len(num_cols)} numerical and {len(cat_cols)} categorical columns in final_data')

print(df_final[num_cols].describe())
print((df_final.assign(**{
  col:df_final[col].astype("object")
  for col in cat_cols
}).describe(include="object")))

import missingno as msno

print(msno.matrix(df_final))
print(msno.heatmap(df_final))
msno.dendrogram(df_final)

def numerical_analysis(dataframe, column_name, cat_col=None, bins="auto"):
    # create the figure
    fig = plt.figure(figsize=(15,10))
    # generate the layout
    grid = GridSpec(nrows=2, ncols=2, figure=fig)
    # set subplots
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])
    # plot the kdeplot
    sns.kdeplot(data=dataframe, x=column_name,hue=cat_col, ax=ax1)
    # plot the boxplot
    sns.boxplot(data=dataframe, x=column_name,hue=cat_col, ax=ax2)
    # plot the histogram
    sns.histplot(data=dataframe, x=column_name,bins=bins,hue=cat_col,kde=True, ax=ax3)
    plt.tight_layout()
    plt.show()


def numerical_categorical_analysis(dataframe, cat_column_1, num_column):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15,7.5))
    # plot the barplot
    sns.barplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[0])
    # plot the boxplot
    sns.boxplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax1[1])
    # plot violin plot
    sns.violinplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[0])
    # plot strip plot
    sns.stripplot(data=dataframe, x=cat_column_1, y=num_column, ax=ax2[1])
    plt.tight_layout()
    plt.show()


def categorical_analysis(dataframe, column_name):
    # print the values counts of categories
    print(
        pd.DataFrame({
            "Count": (
                dataframe[column_name]
                .value_counts()),
            "Percentage": (
                dataframe[column_name]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
                .astype("str")
                .add("%")
                )
        })
    )
    print("*" * 50)
    # get unique categories
    unique_categories = dataframe[column_name].unique().tolist()
    number_of_categories = dataframe[column_name].nunique()
    print(f"The unique categories in {column_name} column are {unique_categories}")
    print("*" * 50)
    print(f"The number of categories in {column_name} column are {number_of_categories}")
    # plot countplot
    sns.countplot(data=dataframe, x=column_name)
    plt.xticks(rotation=45)
    plt.show()


def multivariate_analysis(dataframe, num_column, cat_column_1, cat_column_2):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15,7.5))
    # plot the barplot
    sns.barplot(data=dataframe, x=cat_column_1,
                y=num_column,hue=cat_column_2, ax=ax1[0])
    # plot the boxplot
    sns.boxplot(data=dataframe, x=cat_column_1,
                y=num_column,hue=cat_column_2, gap=0.1, ax=ax1[1])
    # plot violin plot
    sns.violinplot(data=dataframe, x=cat_column_1, gap=0.1,
                   y=num_column,hue=cat_column_2, ax=ax2[0])
    # plot strip plot
    sns.stripplot(data=dataframe, x=cat_column_1,
                  y=num_column,hue=cat_column_2,dodge=True,ax=ax2[1])
    plt.tight_layout()
    plt.show()

def chi_2_test(dataframe, col1, col2, alpha= 0.05):
    data = (
        dataframe.loc[:, [col1, col2]]
        .dropna()
    )
    # create contingency table
    contingency_table = pd.crosstab(data[col1], data[col2])
    # perform chi-squared test
    _, p_val, _, _ = chi2_contingency(contingency_table)
    print(p_val)
    if p_val <= alpha:
        print(f"Reject the null hypothesis. There is a significant association between {col1} and {col2}.")
    else:
        print(f"Fail to reject the null hypothesis. There is no significant association between {col1} and {col2}.")

def anova_test(dataframe, num_col, cat_col, alpha=0.05):
    data = (
        dataframe.loc[:, [num_col, cat_col]]
        .dropna()
    )
    cat_group = data.groupby(cat_col)
    groups = [group[num_col].values for _, group in cat_group]
    f_stat, p_val = f_oneway(*groups)
    print(p_val)
    if p_val <= alpha:
        print(f"Reject the null hypothesis. There is a significant relationship between {num_col} and {cat_col}.")
    else:
        print(f"Fail to reject the null hypothesis. There is no significant relationship between {num_col} and {cat_col}.")

def test_for_normality(dataframe, column_name, alpha=0.05):
    data = dataframe[column_name]
    print("Jarque Bera Test for Normality")
    _, p_val = jarque_bera(data)
    print(p_val)
    if p_val <= alpha:
        print(f"Reject the null hypothesis. The data is not normally distributed.")
    else:
        print(f"Fail to reject the null hypothesis. The data is normally distributed.",end="\n\n")

numerical_analysis(df_final,'time_taken',bins=10)

probplot(df_final['time_taken'],plot=plt)
plt.show()

test_for_normality(df_final,'time_taken')

# check out the rows where data is acting as outlier

target_25_per, target_75_per = np.percentile(df_final['time_taken'], [25, 75])
iqr = target_75_per - target_25_per

upper_bound = target_75_per + (1.5 * iqr)

df_final.loc[(df_final['time_taken'] > upper_bound),"traffic"].value_counts()

df_final.loc[(df_final['time_taken'] > upper_bound),"weather"].value_counts()

# average distances
avg_distance = df_final.loc[:,"distance"].mean()
avg_distance_extreme = df_final.loc[(df_final['time_taken'] > upper_bound),"distance"].mean()

print(avg_distance, avg_distance_extreme)

# fix traget column using transformation

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')

df_final['time_taken_pt'] = pt.fit_transform(df_final[['time_taken']])

numerical_analysis(df_final, "time_taken_pt", bins=10)

# plot QQ plot for the target after transformation

probplot(df_final['time_taken_pt'], plot=plt)
plt.show()

# datatype of rider id

df_final['rider_id'].dtype

df_final[["rider_id","age","ratings"]]

# sample of data

rider_id_group = df_final[["rider_id","age","ratings"]].groupby('rider_id')
rider_id_group.head(5).sort_values('rider_id')

# check for duplicates

df_final[["rider_id","age","ratings"]].dropna().duplicated(keep=False).sum()

# filter the duplicates

(
    df_final
    .loc[(df_final[["rider_id","age","ratings"]].duplicated(keep=False)),["rider_id","age","ratings"]]
    .dropna()
    .sort_values(["rider_id"])
)

# data type of age column

df_final['age'].dtype

# statistical summary

df_final['age'].describe()

# numerical analysis for age

numerical_analysis(df_final, 'age',bins=20)
# relationship between target and age

sns.scatterplot(data=df_final, x='age', y='time_taken')
plt.show()
sns.scatterplot(data=df_final, x='age', y='time_taken', hue="vehicle_condition")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
plt.show()

# preferences of vehicle type based on age

sns.stripplot(df_final,x='type_of_vehicle',y='age')

# data type of rating column

df_final['ratings'].dtype

# statistical summary

df_final['ratings'].describe()

# numerical analysis

numerical_analysis(df_final, 'ratings',bins=5)

# does ratings affect delivery time

sns.scatterplot(data=df_final, x='ratings', y='time_taken')
plt.show()

# does ratings get affected by vehicle type

numerical_categorical_analysis(df_final, 'vehicle_condition', 'ratings')

(
    df_final[["ratings","vehicle_condition"]]
    .loc[df_final["vehicle_condition"]==3,"ratings"]
    .value_counts(dropna=False)
)

# does type of vehicle affects ratings

numerical_categorical_analysis(df_final, 'type_of_vehicle', 'ratings')

# festvals and rider ratings

numerical_categorical_analysis(df_final, 'festival', 'ratings')
df_final.columns[3:7].tolist() + ["city_name"]
# location subset

location_subset = df_final.loc[:,df_final.columns[3:7].tolist() + ["city_name"]]

print(location_subset)

print(location_subset.dtypes)

# drop missing values

location_subset.dropna(inplace=True)

print(location_subset)

# plot deliveries on map

delivery_df = pd.DataFrame({
    'latitude': location_subset['delivery_latitude'],
    'longitude': location_subset['delivery_longitude'],
    "city_name": location_subset["city_name"]
})


# Create a map using Plotly's scatter_mapbox
fig = px.scatter_map(
    delivery_df,
    lat='latitude',
    lon='longitude',
    title="Delivery Points",
    hover_name="city_name"
)
# fig.update_layout(map)

# Update the layout for the map of India
# fig.update_layout(
#     mapbox_style="carto-positron",
#     mapbox_center={"lat": 20.5937, "lon": 78.9629},  # Centered over India
#     mapbox_zoom=3,
# )

# Show the plot
fig.show()

df_final.columns

df_final.filter(like="order")

# order date columns

order_date_subset = df_final.loc[:,["order_date","order_day","order_month","order_day_of_week","is_weekend","festival"]]

order_date_subset

# analysis between day of week and target

numerical_categorical_analysis(df_final, "order_day_of_week", "time_taken")

# does having a weekend affects target

numerical_categorical_analysis(df_final, "is_weekend", "time_taken")

# do weekends have an impact on traffic

chi_2_test(df_final, "is_weekend", "traffic")

# festivals and target analysis

numerical_categorical_analysis(df_final, "festival", "time_taken")

# do festival affect traffic

chi_2_test(df_final, "festival", "traffic")

df_final.pivot_table(index="traffic",columns="festival",values="time_taken",aggfunc="mean")

# does a weekend and a festival combined have an effect on delivery times

multivariate_analysis(df_final, "time_taken", "is_weekend", "festival")

df_final.columns

# time related columns

time_subset = df_final.loc[:,["order_time_hour","order_time_of_day","pickup_time_minutes"]]

time_subset

# does time of day affects delivery times

numerical_categorical_analysis(df_final, "order_time_of_day", "time_taken")

# anova test

anova_test(df_final, "time_taken", "order_time_of_day")

# Top 5 times(hrs) of the day in which customers  order the most

df_final["order_time_hour"].value_counts().head(5)

# categorical analysis on order_time_hour

categorical_analysis(df_final, "order_time_hour")

# categorical analysis on time of day

categorical_analysis(df_final, "order_time_of_day")

# pickup_time datatype

df_final['pickup_time_minutes'].dtype

# relationship between pickup time and delivery time

sns.scatterplot(df_final,x="pickup_time_minutes",y="time_taken")
plt.show()

# pickup time categorical analysis

categorical_analysis(df_final, "pickup_time_minutes")

# does pickup time have an effect on delivery time

numerical_categorical_analysis(df_final, "pickup_time_minutes", "time_taken")

# hypothesis testing to prove point

anova_test(df_final, "time_taken", "pickup_time_minutes")

# datatype of traffic column

df_final['traffic'].dtype

# categorical analysis on traffic

categorical_analysis(df_final, "traffic")

# does traffic depends on type of city

chi_2_test(df_final, "traffic", "city_type")
# does traffic depends on city

chi_2_test(df_final, "traffic", "city_name")

# does traffic affects delivery times

numerical_categorical_analysis(df_final, "traffic", "time_taken")
# hypothesis test on does traffic affects delivery times

anova_test(df_final, "time_taken", "traffic")
# are some vehicle types more suitable in traffic than others

multivariate_analysis(df_final, "time_taken", "traffic", "type_of_vehicle")

# does vehicle condition in traffic situations affects delivery times

multivariate_analysis(df_final, "time_taken", "traffic", "vehicle_condition")

multivariate_analysis(df_final, "time_taken", "festival", "vehicle_condition")

# does multiple delivereis affect delivery times

numerical_categorical_analysis(df_final, "multiple_deliveries", "time_taken")

# hypothesis test

anova_test(df_final, "time_taken", "multiple_deliveries")

# do multiple deliveries are of longer distances

numerical_categorical_analysis(df_final, "multiple_deliveries", "distance")

# data type of weather column

df_final['weather'].dtype

# categorical analysis on type of weather

categorical_analysis(df_final, "weather")

# does weather affect delivery times

numerical_categorical_analysis(df_final, "weather", "time_taken")

# hypothesis test

anova_test(df_final, "time_taken", "weather")
# does the weather affects traffic

chi_2_test(df_final, "weather", "traffic")

# delivery times based on weather and traffic

multivariate_analysis(df_final, "time_taken", "weather", "traffic")

# pivot table

df_final.pivot_table(index="weather",columns="traffic",values="time_taken",aggfunc="mean")

# categorical analysis on vehicle condition

categorical_analysis(df_final, "vehicle_condition")

# does vehicle condition affect delivery times

numerical_categorical_analysis(df_final, "vehicle_condition", "time_taken")

# anova test

anova_test(df_final, "time_taken", "vehicle_condition")
# analysis on vehicle type

categorical_analysis(df_final, "type_of_vehicle")
# does the type of vehicle affects delivery time

numerical_categorical_analysis(df_final, "type_of_vehicle", "time_taken")

# vehicle condition and type

multivariate_analysis(df_final, "time_taken", "vehicle_condition", "type_of_vehicle")

# is their a relation between vehicle type and conditions

chi_2_test(df_final, "type_of_vehicle", "vehicle_condition")

# type of order dtype

df_final['type_of_order'].dtype

# analysis on type of order

categorical_analysis(df_final, "type_of_order")

# does order type have have effect on delivery times

numerical_categorical_analysis(df_final, "type_of_order", "time_taken")

# hypothesis test

anova_test(df_final, "time_taken", "type_of_order")
# contingency table

pd.crosstab(df_final["type_of_order"],df_final["is_weekend"])

# does type of order have an effect over pickup time

chi_2_test(df_final,"pickup_time_minutes","type_of_order")

# does order type has an effect on ratings

numerical_categorical_analysis(df_final, "type_of_order", "ratings")

# is their a relationship between weekends and type of order

chi_2_test(df_final, "is_weekend", "type_of_order")

# is their a relationship between festivals and type of order

chi_2_test(df_final, "festival", "type_of_order")

# categorical analysis on city name

categorical_analysis(df_final, "city_name")

# does a city affects delivery times

numerical_categorical_analysis(df_final, "city_name", "time_taken")

# categorical analysis on city type

categorical_analysis(df_final, "city_type")

# does city types affect delivery times

numerical_categorical_analysis(df_final, "city_type", "time_taken")

# hypothesis test

anova_test(df_final, "time_taken", "city_type")

# does city type affects rider ratings

numerical_categorical_analysis(df_final, "city_type", "ratings")
# city type, type of vehicle and delivery time analysis

multivariate_analysis(df_final, "time_taken", "city_type", "type_of_vehicle")

# numerical analysis of distance

numerical_analysis(df_final, "distance",bins=10)

# relationship between distance and delivery times

sns.scatterplot(df_final,x="distance",y="time_taken")
plt.show()

# corr

df_final[["distance","time_taken"]].corr()

# vehicle type and distance analysis

numerical_categorical_analysis(df_final, "type_of_vehicle", "distance")

# do riders cover more distances during festivals

numerical_categorical_analysis(df_final, "distance", "festival")

# distance and type of vehicle

numerical_categorical_analysis(df_final, "distance", "type_of_vehicle")