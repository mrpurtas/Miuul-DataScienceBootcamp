import pandas as pd
# Standard libraries for data manipulation and visualization
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Adjusting display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score, KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
# Preprocessing and encoding tools
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Machine learning models from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Advanced ensemble models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Model evaluation and hyperparameter tuning tools
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier

# Tools for handling imbalanced data
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Import joblib for saving/loading models
import joblib
import scipy.stats

import warnings
warnings.filterwarnings("ignore")


# İlk olarak 'listings.csv' ve 'listings2.csv' dosyalarını yükleyelim ve içeriğini inceleyelim
listings_path = '/mnt/data/listings.csv'
listings2_path = '/mnt/data/listings2.csv'

df_listings = pd.read_csv("datasets/listings.csv")
df_listings2 = pd.read_csv("datasets/listings2.csv")

# Her iki DataFrame'in ilk birkaç satırına bakalım
df_listings_head = df_listings.head()
df_listings2_head = df_listings2.head()

df_listings_head, df_listings2_head

# Öncelikle ortak sütunları belirleyelim
common_columns = df_listings.columns.intersection(df_listings2.columns)

# Ortak sütunlar üzerinden iki DataFrame'i birleştirelim
df_combined_listings = pd.concat([df_listings[common_columns], df_listings2[common_columns]], ignore_index=True)

# Birleştirilmiş veri setinin ilk birkaç satırına bakalım
df_combined_listings.head()




# 'neighbourhoods.csv' dosyasını yükleyelim ve içeriğini inceleyelim
neighbourhoods_path = '/mnt/data/neighbourhoods.csv'
df_neighbourhoods = pd.read_csv("datasets/neighbourhoods.csv")

# İlk birkaç satırına bakalım
df_neighbourhoods.head()


import geopandas as gpd

# 'neighbourhoods.geojson' dosyasını yükleyelim ve içeriğini inceleyelim
neighbourhoods_geojson_path = '/mnt/data/neighbourhoods.geojson'
df_neighbourhoods_geo = gpd.read_file("datasets/neighbourhoods.geojson")

# İlk birkaç satırına bakalım
df_neighbourhoods_geo.head()

# 'reviews.csv' dosyasını yükleyelim ve içeriğini inceleyelim
reviews_path = '/mnt/data/reviews (1).csv'
df_reviews = pd.read_csv("datasets/reviews.csv")

# İlk birkaç satırına bakalım
df_reviews.head()

import datetime as dt

# Her ilan için toplam yorum sayısını hesaplayalım
total_reviews = df_reviews.groupby('listing_id').size().reset_index(name='total_reviews')

# Her ilan için en son yorum tarihini bulalım
latest_review = df_reviews.groupby('listing_id')['date'].max().reset_index()
latest_review['date'] = pd.to_datetime(latest_review['date'])
latest_review['days_since_last_review'] = (pd.Timestamp.now() - latest_review['date']).dt.days

# Bu bilgileri birleştirelim
review_features = pd.merge(total_reviews, latest_review, on='listing_id')

# Bu bilgileri 'listings' veri setiyle birleştirelim
# Öncelikle 'id' sütununu 'listing_id' ile eşleşecek şekilde değiştirelim
df_combined_listings.rename(columns={'id': 'listing_id'}, inplace=True)

# İlan bilgileri ile yorum özelliklerini birleştirelim
df_combined_listings = pd.merge(df_combined_listings, review_features, on='listing_id', how='left')

# Eksik yorum bilgilerini dolduralım (ilanlar için hiç yorum yapılmamış olabilir)
df_combined_listings['total_reviews'].fillna(0, inplace=True)
df_combined_listings['days_since_last_review'].fillna(df_combined_listings['days_since_last_review'].max(), inplace=True)

# Birleştirilmiş veri setinin ilk birkaç satırını görelim
df_combined_listings.head()
df_combined_listings.shape

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Initial Data Exploration
# This function, check_df(df), is likely designed for initial exploration of the dataframe 'df'.
# It could include operations like checking for missing values, data types, and providing a quick overview of the dataset.
check_df(df_combined_listings)


df_combined_listings.to_csv("/Users/mrpurtas/Desktop/combined_listings.csv", index=False)
df_combined_listings = pd.read_csv("datasets/combined_listings.csv")

cat_cols, num_cols, cat_but_car = grab_col_names(df_combined_listings, cat_th=10, car_th=20)

df_combined_listings[num_cols].describe().T

correlation_matrix(df_combined_listings, num_cols)

# Eksik verileri kontrol edelim
missing_values = df_combined_listings.isnull().sum()
missing_values[missing_values > 0]

"""Out[6]: 
neighbourhood        30228
last_review          39674
license              95036
reviews_per_month    39674
date                 39674
dtype: int64"""


# Eksik verileri işleyelim

# 'neighbourhood' sütunundaki eksik değerleri 'Bilinmiyor' ile doldur
df_combined_listings['neighbourhood'] = df_combined_listings['neighbourhood'].fillna('Bilinmiyor')

# 'last_review', 'reviews_per_month', 'date', 'license' sütunlarını kaldır
df_combined_listings = df_combined_listings.drop(columns=['last_review', 'reviews_per_month', 'date', 'license'])

# Yeni veri çerçevesinin ilk beş satırını gösterelim
df_combined_listings.head()

# 'price' sütununu sayısal bir formata dönüştürelim

# Para birimi işaretlerini ve virgülleri kaldıralım ve float'a dönüştürelim
df_combined_listings['price'] = df_combined_listings['price'].replace('[\$,]', '', regex=True).astype(float)

# Yeniden ilk beş satırı gösterelim
df_combined_listings.head()


def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "Price":
      print(col, check_outlier(df_combined_listings, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "Price" :
        replace_with_thresholds(df_combined_listings,col)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df_combined_listings)


cat_cols, cat_but_car, num_cols = grab_col_names(df_combined_listings)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df_combined_listings.columns if df_combined_listings[col].dtypes == "O" and len(df_combined_listings[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df_combined_listings = one_hot_encoder(df_combined_listings, cat_cols, drop_first=True)

df_combined_listings.columns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Bağımlı ve bağımsız değişkenleri ayıralım
X = df_combined_listings[['latitude', 'longitude', 'minimum_nights', 'availability_365', 'number_of_reviews',
        'number_of_reviews_ltm', 'calculated_host_listings_count', 'total_reviews',
        'days_since_last_review', 'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room']]
y = df_combined_listings['price']

# Veri setini eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lineer regresyon modeli oluşturalım
lr_model = LinearRegression()

# Modeli eğitim verisiyle eğitelim
lr_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapalım
y_pred = lr_model.predict(X_test)

# Performans metriklerini hesaplayalım
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

df_combined_listings.head()

######################################################################################
import pandas as pd

# Yeni yüklenen dosyaların yollarını tanımlama
listings2_path = '/mnt/data/listings2.csv'
reviews_path = '/mnt/data/reviews (1).csv'

# Dosyaları yeniden yükleyelim
listings2_df = pd.read_csv("datasets/listings2.csv")
reviews_df = pd.read_csv("datasets/reviews (1).csv")

# Veri tiplerini kontrol edelim
listings_id_dtype = listings2_df['id'].dtype
reviews_listing_id_dtype = reviews_df['listing_id'].dtype

listings_id_dtype, reviews_listing_id_dtype

# Reviews veri setindeki her bir listing için toplam yorum sayısını ve son yorum tarihini hesaplama
reviews_aggregated = reviews_df.groupby('listing_id').agg(
    total_reviews=('date', 'count'),
    last_review_date=('date', 'max')
).reset_index()

# Listings veri seti ile birleştirme
listings_reviews_combined = pd.merge(listings2_df, reviews_aggregated, how='left', left_on='id', right_on='listing_id')

# Birleştirilmiş veri setinin ilk birkaç satırını görüntüleme
listings_reviews_combined.head()


# neighbourhoods.csv dosyasını yükleme
neighbourhoods_df = pd.read_csv('datasets/neighbourhoods.csv')

# neighbourhoods veri setini listings_reviews_combined veri seti ile birleştirme
combined_df = pd.merge(listings_reviews_combined, neighbourhoods_df, on='neighbourhood', how='left')

# Birleştirilmiş veri setinin ilk birkaç satırını görüntüleme
combined_df.head()


# Eksik veri kontrolü
missing_values = combined_df.isnull().sum()

# Eksik veri oranlarını hesaplama
missing_values_percentage = (missing_values / len(combined_df)) * 100

missing_values_percentage.sort_values(ascending=False)


# Tamamen boş olan sütunları kaldırma
combined_df.drop(['neighbourhood_group_x', 'neighbourhood_group_y', 'license'], axis=1, inplace=True)

# Yorumlarla ilgili eksik verileri işleme
# Yorum yapılmamış evler için total_reviews sütununu 0 ile doldurma
combined_df['total_reviews'].fillna(0, inplace=True)

# Yorum yapılmamış evler için last_review_date ve reviews_per_month sütunlarını uygun bir şekilde işleme
# Bu sütunlar için NaN değerlerin yorum yapılmamış olduğunu gösterdiği varsayılır
combined_df['last_review_date'].fillna("No Reviews", inplace=True)
combined_df['reviews_per_month'].fillna(0, inplace=True)

# Eksik verilerin işlendikten sonraki durumunu kontrol etme
combined_df.isnull().sum()

# listing_id sütununu kaldırma (id sütunu ile aynı bilgiyi taşıyor)
combined_df.drop(columns=['listing_id'], inplace=True)

# last_review ve last_review_date için eksik değerleri 'No Reviews' ile doldurma
combined_df['last_review'].fillna('No Reviews', inplace=True)
combined_df['last_review_date'].fillna('No Reviews', inplace=True)

# Eksik verilerin doldurulduktan sonraki durumu kontrol
missing_data_final = combined_df.isnull().sum()
missing_data_final[missing_data_final > 0]

# Potansiyel olarak gereksiz sütunları kaldırma
columns_to_remove = ['name', 'host_name']
cleaned_df = combined_df.drop(columns=columns_to_remove)

# Kaldırılan sütunlardan sonra veri setinin ilk birkaç satırını kontrol etme
cleaned_df.head()


cleaned_df.to_csv("/Users/mrpurtas/Desktop/cleaned_df.csv", index=False)
cleaned_df = pd.read_csv("datasets/cleaned_df.csv")


cat_cols, num_cols, cat_but_car = grab_col_names(cleaned_df, cat_th=10, car_th=20)



# Sayısal değişkenler için özet istatistiklerin hesaplanması
numeric_summary = cleaned_df[num_cols].describe()
numeric_summary

import numpy as np

# Log dönüşümü uygulanacak değişkenler
log_transform_columns = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                         'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm', 'total_reviews']

# Log dönüşümü (log1p kullanılıyor çünkü sıfır değerleri log dönüşümünde sorun yaratır)
for col in log_transform_columns:
    cleaned_df[col + '_log'] = np.log1p(cleaned_df[col])

# Log dönüşümü uygulandıktan sonraki veri setinin ilk birkaç satırını kontrol etme
cleaned_df.head()
cleaned_df.info()

cleaned_df['last_review'] = pd.to_datetime(cleaned_df['last_review'], errors='coerce')
cleaned_df['last_review_date'] = pd.to_datetime(cleaned_df['last_review_date'], errors='coerce')



# One-Hot Encoding İşlemi
# cat_cols değişkenlerimiz içerisinde hem glikoz seviyesi hem de hedef değişkenimiz bulunduğu için bunların olmadığı ve ohe yapacağımız değişkenleri seçiyoruz
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in cleaned_df.columns if 10 >= cleaned_df[col].nunique() > 2]
cleaned_df = one_hot_encoder(cleaned_df, ohe_cols, drop_first=True)

cleaned_df.head()

cleaned_df = pd.get_dummies(cleaned_df, columns=['neighbourhood'], drop_first=True)


num_cols = [col for col in num_cols if col not in ['id', 'host_id', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm', 'total_reviews', 'last_review_date', 'price_log']]

from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
cleaned_df[num_cols] = rs.fit_transform(cleaned_df[num_cols])

price_log
minimum_nights_log
number_of_reviews_log
reviews_per_month_log
calculated_host_listings_count_log
availability_365_log
number_of_reviews_ltm_log
total_reviews_log

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if check_outlier(cleaned_df, col):
        print(f"Outliers detected in {col}. Replacing outliers with thresholds.")
        replace_with_thresholds(cleaned_df, col)
    else:
        print(f"No outliers detected in {col}.")

cleaned_df.head()

corr = cleaned_df[num_cols].corr()

# Set figure size and create a heatmap with annotations
plt.figure(figsize=(12, 12))
sns.heatmap(corr, cmap="RdBu_r", annot=True, fmt=".2f", linewidths=0.5)

# Add labels and title
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")

# Show the plot
plt.show()

cleaned_df.columns


cleaned_df['day_of_week'] = cleaned_df['last_review'].dt.dayofweek
cleaned_df['month'] = cleaned_df['last_review'].dt.month

from datetime import datetime
cleaned_df['days_since_last_review'] = (datetime.now() - cleaned_df['last_review_date']).dt.days

#'number_of_reviews' ve 'reviews_per_month' sütunlarının toplamını hesaplama:
cleaned_df['total_reviews'] = cleaned_df['number_of_reviews'] + (cleaned_df['reviews_per_month'] * 12)

#İnceleme puanlarının ortalama değerini hesaplama:
cleaned_df['average_review_score'] = (cleaned_df['review_scores_rating'] + cleaned_df['review_scores_accuracy'] + cleaned_df['review_scores_cleanliness'] + cleaned_df['review_scores_checkin'] + cleaned_df['review_scores_communication'] + cleaned_df['review_scores_location'] + cleaned_df['review_scores_value']) / 7

#Ev sahibinin yönettiği liste sayısını kullanma:
cleaned_df['host_listings_count'] = cleaned_df.groupby('host_id')['host_id'].transform('count')

"""'minimum_nights' ve 'availability_365' sütunlarını kullanarak rezervasyon esnekliğini hesaplama:"""
cleaned_df['booking_flexibility'] = cleaned_df['availability_365'] / cleaned_df['minimum_nights']


#Fiyatın karekökünü almak:
cleaned_df['price_sqrt'] = cleaned_df['price'].apply(lambda x: x**0.5)


cleaned_df.columns
cleaned_df.head()


cat_cols, cat_but_car, num_cols = grab_col_names(cleaned_df)




def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cleaned_df = one_hot_encoder(cleaned_df, ["day_of_week"], drop_first=True)

cleaned_df['month'] = cleaned_df['month'].fillna(0)
cleaned_df['days_since_last_review'] = cleaned_df['days_since_last_review'].fillna(0)


# Bağımsız değişkenler (log dönüşümlü değişkenler dahil)
X = cleaned_df[['latitude', 'longitude', 'minimum_nights_log', 'number_of_reviews_log',
        'reviews_per_month_log', 'calculated_host_listings_count_log', 'availability_365_log',
        'number_of_reviews_ltm_log', #'total_reviews_log',
        'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room',
        # Tüm mahalleler
        'neighbourhood_Arnavutkoy', 'neighbourhood_Atasehir', 'neighbourhood_Avcilar',
        'neighbourhood_Bagcilar', 'neighbourhood_Bahcelievler', 'neighbourhood_Bakirkoy',
        'neighbourhood_Basaksehir', 'neighbourhood_Bayrampasa', 'neighbourhood_Besiktas',
        'neighbourhood_Beykoz', 'neighbourhood_Beylikduzu', 'neighbourhood_Beyoglu',
        'neighbourhood_Buyukcekmece', 'neighbourhood_Catalca', 'neighbourhood_Cekmekoy',
        'neighbourhood_Esenler', 'neighbourhood_Esenyurt', 'neighbourhood_Eyup',
        'neighbourhood_Fatih', 'neighbourhood_Gaziosmanpasa', 'neighbourhood_Gungoren',
        'neighbourhood_Kadikoy', 'neighbourhood_Kagithane', 'neighbourhood_Kartal',
        'neighbourhood_Kucukcekmece', 'neighbourhood_Maltepe', 'neighbourhood_Pendik',
        'neighbourhood_Sancaktepe', 'neighbourhood_Sariyer', 'neighbourhood_Sile',
        'neighbourhood_Silivri', 'neighbourhood_Sisli', 'neighbourhood_Sultanbeyli',
        'neighbourhood_Sultangazi', 'neighbourhood_Tuzla', 'neighbourhood_Umraniye',
        'neighbourhood_Uskudar', 'neighbourhood_Zeytinburnu',
        'month', 'days_since_last_review', 'host_listings_count',
        'booking_flexibility', 'day_of_week_1.0', 'day_of_week_2.0',
        'day_of_week_3.0', 'day_of_week_4.0', 'day_of_week_5.0', 'day_of_week_6.0']]
y = cleaned_df['price_log']

def base_models_regression(X, y, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']):
    print("Base Models for Regression....")

    regressors = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.1)),
        ('Decision Tree Regressor', DecisionTreeRegressor()),
        ('Random Forest Regressor', RandomForestRegressor(n_estimators=100)),
        ('Gradient Boosting Regressor', GradientBoostingRegressor()),
        ('XGBoost Regressor', XGBRegressor()),
        ('LightGBM Regressor', LGBMRegressor())
    ]

    for name, regressor in regressors:
        print(f"Evaluating {name} Regressor:")
        for metric in scoring:
            scores = cross_val_score(regressor, X, y, cv=3, scoring=metric)
            mean_score = round(scores.mean(), 4)
            std_score = round(scores.std(), 4)
            print(f"{metric}: {mean_score} (Std: {std_score}) ({name})")

# Fonksiyonun kullanımı
base_models_regression(X, y)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


cleaned_df.to_csv("/Users/mrpurtas/Desktop/cleaned_df.csv", index=False)

"""Base Models for Regression....
Evaluating Linear Regression Regressor:
neg_mean_squared_error: -0.5298 (Std: 0.073) (Linear Regression)
neg_mean_absolute_error: -0.5434 (Std: 0.0273) (Linear Regression)
r2: 0.2296 (Std: 0.0352) (Linear Regression)
Evaluating Ridge Regressor:
neg_mean_squared_error: -0.5298 (Std: 0.0727) (Ridge)
neg_mean_absolute_error: -0.5435 (Std: 0.027) (Ridge)
r2: 0.2294 (Std: 0.0349) (Ridge)
Evaluating Lasso Regressor:
neg_mean_squared_error: -0.6528 (Std: 0.0779) (Lasso)
neg_mean_absolute_error: -0.6038 (Std: 0.0306) (Lasso)
r2: 0.0496 (Std: 0.0152) (Lasso)
Evaluating Decision Tree Regressor Regressor:
neg_mean_squared_error: -0.8639 (Std: 0.1108) (Decision Tree Regressor)
neg_mean_absolute_error: -0.6665 (Std: 0.0372) (Decision Tree Regressor)
r2: -0.2668 (Std: 0.0803) (Decision Tree Regressor)
####Evaluating Random Forest Regressor Regressor:
neg_mean_squared_error: -0.459 (Std: 0.0818) (Random Forest Regressor)
neg_mean_absolute_error: -0.4817 (Std: 0.0267) (Random Forest Regressor)
r2: 0.3336 (Std: 0.0531) (Random Forest Regressor)
#######Evaluating Gradient Boosting Regressor Regressor:
neg_mean_squared_error: -0.4722 (Std: 0.0691) (Gradient Boosting Regressor)
neg_mean_absolute_error: -0.5015 (Std: 0.0236) (Gradient Boosting Regressor)
r2: 0.3139 (Std: 0.0332) (Gradient Boosting Regressor)
#####Evaluating XGBoost Regressor Regressor:
neg_mean_squared_error: -0.4506 (Std: 0.0802) (XGBoost Regressor)
neg_mean_absolute_error: -0.4795 (Std: 0.0293) (XGBoost Regressor)
r2: 0.3475 (Std: 0.0509) (XGBoost Regressor)
#####Evaluating LightGBM Regressor Regressor:
neg_mean_squared_error: -0.4375 (Std: 0.0671) (LightGBM Regressor)
neg_mean_absolute_error: -0.474 (Std: 0.0206) (LightGBM Regressor)
r2: 0.3651 (Std: 0.0325) (LightGBM Regressor)"""



# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitin
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = rf_model.predict(X_test)

# Modelin performansını değerlendirin
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Tahmin hatalarının standart sapmasını hesaplayın
errors = y_test - y_pred
std_dev = np.std(errors)

print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")
print(f"Standard Deviation of Prediction Errors: {std_dev}")
"""Root Mean Squared Error: 0.60021987945361
R^2 Score: 0.4701591243251855
Standard Deviation of Prediction Errors: 0.6000767892108815"""

plot_importance(rf_model, X)

# Eğer y log dönüşümü uygulanmışsa, tahminleri orijinal ölçeğe dönüştürün
# y_pred_exp = np.exp(y_pred)
# print(y_pred_exp)

# LightGBM modelini eğitin
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = lgbm_model.predict(X_test)

# Modelin performansını değerlendirin
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Tahmin hatalarının standart sapmasını hesaplayın
errors = y_test - y_pred
std_dev = np.std(errors)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_model, X)
#Root Mean Squared Error: 0.6170336986581676
#R^2 Score: 0.4400587329623983
#Standard Deviation of Prediction Errors: 0.6170236050210693

# Korelasyon matrisini hesaplayın
corr_matrix = X.corr()

plt.figure(figsize=(20, 20))  # Figür boyutunu artır
sns.set(font_scale=1.25)  # Etiket boyutunu artır
corr_matrix = X.corr()  # Korelasyon matrisini hesapla
sns.heatmap(corr_matrix, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title("Bağımsız Değişkenler Arasındaki Korelasyon", fontsize=20)
plt.xticks(rotation=90)  # X eksenindeki etiketleri 90 derece döndür
plt.yticks(rotation=0)  # Y eksenindeki etiketleri düz tut
plt.tight_layout()  # Çıktıyı düzgün hale getir
plt.show()





from sklearn.model_selection import cross_val_score

def base_models_regression_with_cv(X, y, cv=5):
    print("Base Models with Cross-Validation for Regression....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressors = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.1)),
        ('Decision Tree Regressor', DecisionTreeRegressor()),
        ('Random Forest Regressor', RandomForestRegressor(n_estimators=100)),
        ('Gradient Boosting Regressor', GradientBoostingRegressor()),
        ('XGBoost Regressor', XGBRegressor()),
        ('LightGBM Regressor', LGBMRegressor())
    ]

    for name, regressor in regressors:
        scores = cross_val_score(regressor, X, y, cv=cv, scoring='r2')
        mse_scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_absolute_error')

        print(f"{name} Regressor:")
        print(f"Mean R^2 Score: {np.mean(scores)} (std: {np.std(scores)})")
        print(f"Mean Negative Mean Squared Error: {np.mean(mse_scores)} (std: {np.std(mse_scores)})")
        print(f"Mean Negative Mean Absolute Error: {np.mean(mae_scores)} (std: {np.std(mae_scores)})\n")

# Fonksiyonu kullanma örneği
base_models_cv = base_models_regression_with_cv(X, y, cv=5)

"""Random Forest Regressor Regressor:
Mean R^2 Score: 0.3619039918315422 (std: 0.07023664204983165)
Mean Negative Mean Squared Error: -0.4398890156475514 (std: 0.08302271052066215)
Mean Negative Mean Absolute Error: -0.47010870289431433 (std: 0.031665753867233366)

XGBoost Regressor Regressor:
Mean R^2 Score: 0.37016790450669096 (std: 0.06945994580637402)
Mean Negative Mean Squared Error: -0.4347957440966663 (std: 0.08620594250266676)
Mean Negative Mean Absolute Error: -0.4745667709645887 (std: 0.03861186162062174)

LightGBM Regressor Regressor:
Mean R^2 Score: 0.3840804473541372 (std: 0.04012778948177678)
Mean Negative Mean Squared Error: -0.42406104524875676 (std: 0.06737500199430281)
Mean Negative Mean Absolute Error: -0.4690395982632487 (std: 0.02263523106258463)
"""

"""rf_params = {
    "n_estimators": range(10, 300, 50),
    "max_depth": [None] + list(range(5, 31, 5)),
    "min_samples_split": range(2, 20, 4),
    "min_samples_leaf": range(1, 20, 4),
    "max_features": ['None', 'sqrt', 'log2']
}

# Define hyperparameter grids for XGBoost
xgboost_params = {
    "n_estimators": range(50, 300, 50),
    "max_depth": range(3, 11, 2),
    "learning_rate": np.arange(0.01, 0.3, 0.05),
    "subsample": np.arange(0.5, 1.05, 0.1),
    "colsample_bytree": np.arange(0.1, 1.05, 0.1)
}

# Define hyperparameter grids for LightGBM
lightgbm_params = {
    "num_leaves": range(20, 200, 30),
    "learning_rate": np.arange(0.01, 0.3, 0.05),
    "n_estimators": range(50, 300, 50),
    "subsample": np.arange(0.5, 1.05, 0.1),
    "colsample_bytree": np.arange(0.5, 1.05, 0.1)
}
"""


classifiers = [
    #('LR', LogisticRegression(class_weight='balanced', verbose=0), lr_params),
    # ('KNN', make_pipeline(SMOTE(random_state=42), KNeighborsClassifier())),
    #("SVC", SVC(class_weight='balanced', probability=True), svc_params),
    #("CART", DecisionTreeClassifier(class_weight='balanced'), cart_params),
    ("RF", RandomForestRegressor(), rf_params),
    # ('Adaboost', make_pipeline(SMOTE(random_state=42), AdaBoostClassifier())),
    # ('GBM', make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier())),
    ('XGBoost', XGBRegressor(), xgboost_params),
    ('LightGBM', LGBMRegressor(verbose=-1), lightgbm_params),
    #('NaiveBayes', GaussianNB(), naive_bayes_params) # I have set the prior parameter to [1, 4] because there are 215 samples in class 0 and 46 samples in class 1.
]

def hyperparameter_optimization(X, y, regressors, cv=3, n_iter=100):
    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")

        # Perform hyperparameter optimization using RandomizedSearchCV
        rs_best = RandomizedSearchCV(regressor, params, cv=cv, n_iter=n_iter, n_jobs=-1, verbose=0, random_state=42).fit(X, y)

        # Set the final model with the best hyperparameters
        final_model = regressor.set_params(**rs_best.best_params_)

        # Calculate R^2 and RMSE scores after hyperparameter optimization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"RMSE (After): {rmse}")
        print(f"R^2 Score (After): {r2}")
        print(f"MAE (After): {mae}")

        best_models[name] = final_model
        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")

    return best_models

# Örnek kullanım:
# classifiers parametresini düzeltmek için uygun regresör ve parametre gridlerini burada tanımlayın.
# Örneğin: regressors = [('Ridge', Ridge(), {'alpha': [0.01, 0.1, 1, 10]}), ...]
best_models = hyperparameter_optimization(X, y, classifiers, cv=3, n_iter=100)


rf_params = {
    "n_estimators": [200, 210, 220, 230, 240, 250],
    "max_depth": [None, 20, 25, 30],
    "min_samples_split": [16, 18, 20],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ['sqrt', 'log2']
}

xgboost_params = {
    "n_estimators": [150, 175, 200, 225, 250],
    "max_depth": [7, 8, 9, 10],
    "learning_rate": [0.05, 0.06, 0.07, 0.08],
    "subsample": [0.85, 0.9, 0.95],
    "colsample_bytree": [0.65, 0.7, 0.75]
}

lightgbm_params = {
    "num_leaves": [80, 110, 140, 170],
    "learning_rate": [0.05, 0.06, 0.07, 0.08],
    "n_estimators": [200, 225, 250, 275],
    "subsample": [0.85, 0.9, 0.95],
    "colsample_bytree": [0.45, 0.5, 0.55]
}

"""########## RF ##########
RMSE (After): 0.6121043332904854
R^2 Score (After): 0.44896952638420484
MAE (After): 0.4426234898498117
RF best params: {'n_estimators': 200, 'min_samples_split': 16, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
########## XGBoost ##########
RMSE (After): 0.5907270039286207
R^2 Score (After): 0.48678616226300664
MAE (After): 0.42168694384913513
XGBoost best params: {'subsample': 0.85, 'n_estimators': 225, 'max_depth': 9, 'learning_rate': 0.05, 'colsample_bytree': 0.65}
########## LightGBM ##########
RMSE (After): 0.59357380094627
R^2 Score (After): 0.48182774311692667
MAE (After): 0.42536318761448405
LightGBM best params: {'subsample': 0.85, 'num_leaves': 140, 'n_estimators': 200, 'learning_rate': 0.05, 'colsample_bytree': 0.45}"""










import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


import optuna
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer


def objective(trial, X, y, regressor_class, param_distributions):
    # Optuna trial objesi için hiperparametreleri seç
    kwargs = {}
    for param_name, param_distribution in param_distributions.items():
        if isinstance(param_distribution[0], int):
            kwargs[param_name] = trial.suggest_int(param_name, param_distribution[0], param_distribution[-1])
        elif isinstance(param_distribution[0], float):
            kwargs[param_name] = trial.suggest_float(param_name, param_distribution[0], param_distribution[-1])
        elif isinstance(param_distribution[0], str):
            kwargs[param_name] = trial.suggest_categorical(param_name, param_distribution)

    # Model nesnesini oluştur
    regressor = regressor_class(**kwargs)

    # Belirtilen skorlama metriği ile çapraz doğrulama skorlarını hesapla
    scoring = {
        'r2': make_scorer(r2_score),
        'MSE': make_scorer(mean_squared_error, greater_is_better=False),
        'RMSE': make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
                            greater_is_better=False),
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False)
    }
    cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring)
    mean_mse = -cv_results['test_MSE'].mean()  # Negated to make positive

    return mean_mse  # Optuna optimizes based on MSE

def hyperparameter_optimization_with_optuna(X, y, regressors, cv=3, n_iter=100):
    best_models = {}

    for name, regressor_class, param_distributions in regressors:
        print(f"########## Optimizing {name} ##########")

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X, y, regressor_class, param_distributions), n_trials=n_iter)

        # En iyi parametreleri al ve modeli ayarla
        best_params = study.best_params
        best_regressor = regressor_class(**best_params)
        best_regressor.fit(X, y)

        # Çapraz doğrulama ile performans metriklerini hesapla
        cv_results = cross_validate(best_regressor, X, y, cv=cv, scoring=scoring)
        mean_r2 = cv_results['test_r2'].mean()
        mean_rmse = -cv_results['test_RMSE'].mean()  # Negated to make positive
        mean_mae = -cv_results['test_MAE'].mean()  # Negated to make positive

        print(f"{name} Regressor:")
        print(f"Cross-Validated Mean R^2: {mean_r2}")
        print(f"Cross-Validated Mean RMSE: {mean_rmse}")
        print(f"Cross-Validated Mean MAE: {mean_mae}")
        print(f"Best params: {best_params}\n")

        best_models[name] = best_regressor

    return best_models

# Replace 'regressors' with your list of regressor configurations
# best_modelss = hyperparameter_optimization_with_optuna(X, y, regressors, cv=3, n_iter=100)


# Replace 'regressors' with your list of regressor configurations
best_modelss = hyperparameter_optimization_with_optuna(X, y, regressors, cv=3, n_iter=100)

regressors = [
    #('Linear Regression', LinearRegression(), {}),
    #('Ridge', Ridge(), {'alpha': [1.0]}),
    #('Lasso', Lasso(), {'alpha': [0.1]}),
    #('Decision Tree Regressor', DecisionTreeRegressor(), {}),
    ('Random Forest Regressor', RandomForestRegressor(), rf_params),
    #('Gradient Boosting Regressor', GradientBoostingRegressor(), {}),
    ('XGBoost Regressor', XGBRegressor(), xgboost_params),
    ('LightGBM Regressor', LGBMRegressor(), lightgbm_params)
]
rf_params = {
    "n_estimators": range(100, 1001, 100),
    "max_depth": [None] + list(range(5, 51, 5)),
    "min_samples_split": range(2, 50, 5),
    "min_samples_leaf": range(1, 50, 5),
    "max_features": [None, 'auto', 'sqrt', 'log2']
}


xgboost_params = {
    "n_estimators": range(100, 1001, 100),
    "max_depth": range(3, 51, 3),
    "learning_rate": np.arange(0.01, 0.6, 0.05),
    "subsample": np.arange(0.1, 1.0, 0.1),
    "colsample_bytree": np.arange(0.1, 1.0, 0.1)
}

lightgbm_params = {
    "num_leaves": range(30, 300, 30),
    "learning_rate": np.arange(0.01, 0.6, 0.05),
    "n_estimators": range(100, 1001, 100),
    "subsample": np.arange(0.1, 1.0, 0.1),
    "colsample_bytree": np.arange(0.1, 1.0, 0.1)
}



from sklearn.model_selection import cross_validate, train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
import numpy as np

def hyperparameter_optimization(X, y, regressors, cv=3, n_iter=100):
    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")

        # Perform hyperparameter optimization using RandomizedSearchCV
        rs_best = RandomizedSearchCV(regressor, params, cv=cv, n_iter=n_iter, n_jobs=-1, verbose=1, random_state=42)
        rs_best.fit(X, y)

        # Set the final model with the best hyperparameters
        final_model = regressor.set_params(**rs_best.best_params_)

        # Define the scoring dictionary
        scoring = {
            'r2': make_scorer(r2_score),
            'MSE': make_scorer(mean_squared_error, greater_is_better=False),
            'RMSE': make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False), greater_is_better=False),
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False)
        }

        # Calculate scores after hyperparameter optimization using cross-validation
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)

        # Extract the mean scores and print them
        mean_r2 = cv_results['test_r2'].mean()
        mean_mse = -cv_results['test_MSE'].mean()  # Negated to make positive
        mean_rmse = -cv_results['test_RMSE'].mean()  # Negated to make positive
        mean_mae = -cv_results['test_MAE'].mean()  # Negated to make positive

        print(f"{name} Regressor (After optimization):")
        print(f"Mean R^2: {mean_r2}")
        print(f"Mean MSE: {mean_mse}")
        print(f"Mean RMSE: {mean_rmse}")
        print(f"Mean MAE: {mean_mae}\n")

        # Store the model and its scores in the dictionary
        best_models[name] = {
            'model': final_model,
            'mean_r2': mean_r2,
            'mean_mse': mean_mse,
            'mean_rmse': mean_rmse,
            'mean_mae': mean_mae,
            'best_params': rs_best.best_params_
        }

        print(f"{name} best params: {rs_best.best_params_}\n")

    return best_models

best_models = hyperparameter_optimization(X, y, classifiers, cv=3, n_iter=100)


"""########## RF ##########
Fitting 3 folds for each of 100 candidates, totalling 300 fits
RF Regressor (After optimization):
Mean R^2: 0.35383649917126236
Mean MSE: 0.44489167172020355
Mean RMSE: 0.665270957624929
Mean MAE: 0.4825336898566401
RF best params: {'n_estimators': 230, 'min_samples_split': 18, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}

########## XGBoost ##########
Fitting 3 folds for each of 100 candidates, totalling 300 fits
XGBoost Regressor (After optimization):
Mean R^2: 0.38292408692000784
Mean MSE: 0.42541481445577695
Mean RMSE: 0.6502340845038072
Mean MAE: 0.4650811450228863

XGBoost best params: {'subsample': 0.85, 'n_estimators': 225, 'max_depth': 9, 'learning_rate': 0.05, 'colsample_bytree': 0.65}
########## LightGBM ##########
Fitting 3 folds for each of 100 candidates, totalling 300 fits
LightGBM Regressor (After optimization):
Mean R^2: 0.38702528948859644
Mean MSE: 0.4224290644349428
Mean RMSE: 0.6480603664805602
Mean MAE: 0.4647578422139235
LightGBM best params: {'subsample': 0.85, 'num_leaves': 140, 'n_estimators': 200, 'learning_rate': 0.05, 'colsample_bytree': 0.45}
"""


rf_params = {
    "n_estimators": range(100, 1001, 100),
    "max_depth": [None] + list(range(5, 51, 5)),
    "min_samples_split": range(2, 50, 5),
    "min_samples_leaf": range(1, 50, 5),
    "max_features": [None, 'auto', 'sqrt', 'log2']
}


xgboost_params = {
    "n_estimators": range(100, 1001, 100),
    "max_depth": range(3, 51, 3),
    "learning_rate": np.arange(0.01, 0.6, 0.05),
    "subsample": np.arange(0.1, 1.0, 0.1),
    "colsample_bytree": np.arange(0.1, 1.0, 0.1)
}

lightgbm_params = {
    "num_leaves": range(30, 300, 30),
    "learning_rate": np.arange(0.01, 0.6, 0.05),
    "n_estimators": range(100, 1001, 100),
    "subsample": np.arange(0.1, 1.0, 0.1),
    "colsample_bytree": np.arange(0.1, 1.0, 0.1)
}

"""########## RF ##########
Fitting 3 folds for each of 100 candidates, totalling 300 fits
RF Regressor (After optimization):
Mean R^2: 0.3565003799824427
Mean MSE: 0.4430966219928239
Mean RMSE: 0.663903006830224
Mean MAE: 0.4808617510316542
RF best params: {'n_estimators': 800, 'min_samples_split': 12, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
########## XGBoost ##########
Fitting 3 folds for each of 100 candidates, totalling 300 fits
XGBoost Regressor (After optimization):
Mean R^2: 0.3785411205137872
Mean MSE: 0.4285352820964725
Mean RMSE: 0.6525795964625534
Mean MAE: 0.4680827386549413
XGBoost best params: {'subsample': 0.30000000000000004, 'n_estimators': 600, 'max_depth': 15, 'learning_rate': 0.01, 'colsample_bytree': 0.6000000000000001}
########## LightGBM ##########
Fitting 3 folds for each of 100 candidates, totalling 300 fits
python(25102) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(25103) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
LightGBM Regressor (After optimization):
Mean R^2: 0.38922900271381033
Mean MSE: 0.4209537130639909
Mean RMSE: 0.6468908121627553
Mean MAE: 0.46252893125118066
LightGBM best params: {'subsample': 0.4, 'num_leaves': 210, 'n_estimators': 800, 'learning_rate': 0.01, 'colsample_bytree': 0.5}"""


import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

def rf_objective(trial):
    # Random Forest için parametreleri seç
    rf_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, 100),
        "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(5, 51, 5))),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 45, 5),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 45, 5),
        "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', None])
    }

    # Random Forest modelini oluştur
    model = RandomForestRegressor(**rf_params)

    # Çapraz doğrulama skorunu (R^2) hesapla
    r2_scores = cross_val_score(model, X, y, n_jobs=-1, cv=3, scoring="r2")

    # Ortalama R^2 skorunu döndür
    return np.mean(r2_scores)



############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################

df = pd.read_csv("datasets/listings.csv")
df.head()
# Eksik değerlerin sayısını ve yüzdesini hesaplayalım
missing_values = df.isnull().sum()
missing_percent = (missing_values / df.shape[0]) * 100

# Eksik değer tablosunu oluşturalım
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

# İlk 20 sütunu gösterelim
missing_data.head(20)

# Tüm değerleri eksik olan sütunları kaldıralım
data_cleaned = df.dropna(axis=1, how='all')

missing_values = data_cleaned.isnull().sum()
missing_percent = (missing_values / data_cleaned.shape[0]) * 100

# Eksik değer tablosunu oluşturalım
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

# Belirli bir yüzde üzerinde eksik veri içeren sütunları kaldırmayı düşünebiliriz
# Örneğin, %60'tan fazla eksik verisi olan sütunları kaldıralım
threshold = 60  # yüzde olarak
columns_to_drop = missing_data[missing_data['Percent'] > threshold].index
data_cleaned.drop(columns=columns_to_drop, inplace=True)

# Eksik verileri temizledikten sonra kalan sütunların sayısını kontrol edelim
remaining_columns = data_cleaned.shape[1]
remaining_columns

# Tüm değerleri eksik olan sütunları kaldıralım      days_since_last_review reviews_per_month activity_duration average_monthly_reviews
data_cleaned = df.dropna(axis=1, how='all')


# 'host_neighbourhood' sütununu veri setinden çıkaralım
data_cleaned = data_cleaned.drop('host_neighbourhood', axis=1)

# 'host_neighbourhood' sütununun çıkarıldığını kontrol edelim
"host_neighbourhood" in data_cleaned.columns

data_cleaned = data_cleaned.drop('host_neighbourhood', axis=1)
data_cleaned = data_cleaned.drop('neighbourhood', axis=1)

data_cleaned.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(data_cleaned)

review_columns = ['review_scores_value', 'review_scores_checkin', 'review_scores_location',
                  'review_scores_cleanliness', 'review_scores_communication',
                  'review_scores_accuracy', 'review_scores_rating']

data_cleaned['total_review_score'] = data_cleaned[review_columns].sum(axis=1)

# Her bir review skoru için ağırlıklar (örnek olarak)
weights = {
    'review_scores_value': 2,  # Fiyat/değer oranının önemi yüksek olabilir.
    'review_scores_checkin': 1,  # Check-in sürecinin pürüzsüz olması önemli ancak diğerlerinden daha az.
    'review_scores_location': 3,  # Konum genellikle fiyat için çok önemli bir faktördür.
    'review_scores_cleanliness': 4,  # Temizlik, özellikle konaklama sektöründe, çok önemlidir.
    'review_scores_communication': 1,  # İletişim önemli ancak fiyat üzerinde dolaylı etkisi olabilir.
    'review_scores_accuracy': 2,  # İlanın doğruluğu, müşteri memnuniyeti için önemlidir.
    'review_scores_rating': 5   # Genel puanlama, konukların genel memnuniyetini yansıtır.
}

# Ağırlıklı toplamı hesaplamak için bir fonksiyon
def weighted_review_score(row, weight_dict):
    total_score = 0
    total_weight = sum(weight_dict.values())
    for col, weight in weight_dict.items():
        # Eğer skor NaN ise, bu skoru hesaba katmamak için sıfır puan verilebilir.
        total_score += row[col] * weight if not pd.isna(row[col]) else 0
    return total_score / total_weight

# Ağırlıklı review toplam skoru hesaplama
data_cleaned['weighted_total_review_score'] = data_cleaned.apply(lambda row: weighted_review_score(row, weights), axis=1)


data_cleaned['activity_duration'] = (pd.to_datetime(data_cleaned['last_review']) - pd.to_datetime(data_cleaned['first_review'])).dt.days

data_cleaned['average_monthly_reviews'] = data_cleaned['reviews_per_month'] * (data_cleaned['activity_duration'] / 30)

data_cleaned['days_since_last_review'] = (pd.to_datetime('today') - pd.to_datetime(data_cleaned['last_review'])).dt.days


# Eksik veri içeren satırları çıkarmadan önceki satır sayısını alalım
original_row_count = data_cleaned.shape[0]

# Eksik veri içeren satırları çıkaralım
dff = data_cleaned.dropna(subset=['days_since_last_review', 'reviews_per_month', 'activity_duration', 'average_monthly_reviews'])

# Eksik veri çıkarıldıktan sonraki satır sayısını alalım
new_row_count = dff.shape[0]

# Orjinal ve yeni satır sayıları arasındaki farkı hesaplayalım
row_difference = original_row_count - new_row_count
original_row_count, new_row_count, row_difference


missing_values = dff.isnull().sum()
missing_percent = (missing_values / dff.shape[0]) * 100

# Eksik değer tablosunu oluşturalım
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

dff.head()
dff.columns


import pandas as pd
from datetime import datetime

df = dff.copy()

# 'host_since' tarihinden bugüne kadar geçen süreyi hesaplayarak ev sahibinin platformda ne kadar süredir aktif olduğunu gösteren bir özellik oluşturun
df['host_duration'] = (pd.to_datetime('today') - pd.to_datetime(df['host_since'])).dt.days

# 'host_response_rate' ve 'host_acceptance_rate' değerlerini yüzdelikten sayısal bir forma dönüştürün
dff['host_response_rate'] = dff['host_response_rate'].str.rstrip('%').astype('float') / 100.0
df['host_acceptance_rate'] = df['host_acceptance_rate'].str.rstrip('%').astype('float') / 100.0

# 'host_response_time' sütununu kategorik değişkene dönüştürün
df['host_response_time'] = df['host_response_time'].map({
    'within an hour': 1,
    'within a few hours': 2,
    'within a day': 3,
    'a few days or more': 4
})

# 'host_listings_count' kullanarak ev sahibinin toplam ilan sayısını direkt olarak kullanın veya eşik değere göre kategorize edin
df['is_multi_host'] = df['host_listings_count'] > 1  # 1'den fazla ilanı olan host'lar için True

# 'host_about' metin sütununu analiz ederek, metin uzunluğunu bir özellik olarak kullanın
df['host_about_length'] = df['host_about'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)

# 'host_has_profile_pic' ve 'host_identity_verified' sütunlarını ikili (0 veya 1) özellikler olarak kullanın
df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1, 'f': 0})
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0})


df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})

df['host_verification_count'] = df['host_verifications'].apply(lambda x: len(x.split(',')))

# Örnek ağırlıklar
weights = {
    'host_duration': 0.2,  # Platformdaki süre
    'host_response_rate': 0.1,  # Yanıt oranı
    'host_acceptance_rate': 0.1,  # Kabul oranı
    'is_multi_listing_host': 0.1,  # Birden fazla ilanı olup olmaması
    'host_has_profile_pic': 0.05,  # Profil resmi olup olmaması
    'host_identity_verified': 0.1,  # Kimlik doğrulanmış mı
    'host_about_length': 0.1,  # Ev sahibi hakkında bilgi uzunluğu
    'host_is_superhost': 0.25,  # Superhost statüsü
    'host_verification_count': 0.1  # Doğrulama yöntemleri sayısı
}

# Ağırlıklı toplamı hesaplamak için bir fonksiyon
def calculate_host_activity_score(row, weight_dict):
    activity_score = 0
    for key, weight in weight_dict.items():
        activity_score += row[key] * weight
    return activity_score

df['host_activity_score'] = df.apply(calculate_host_activity_score, axis=1, args=(weights,))



#Ev Sahibi Güvenilirlik Skoru:
df['host_reliability_score'] = (df['host_identity_verified'] + df['host_has_profile_pic'] + df['host_is_superhost']) / 3

#Ev Sahibi Deneyim Skoru:
df['host_experience_score'] = df['host_duration'] * df['host_listings_count']



missing_values = df.isnull().sum()
missing_percent = (missing_values / df.shape[0]) * 100

# Eksik değer tablosunu oluşturalım
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

df.columns

# Eksik 'bedrooms' değerlerini doldurmak için bir fonksiyon
def fill_bedrooms(row, group_data):
    if pd.isna(row['bedrooms']):
        return group_data.get((row['property_type'], row['room_type'], row['accommodates']))
    else:
        return row['bedrooms']

# 'property_type', 'room_type', 'accommodates' kombinasyonlarına göre yatak odası ortalamalarını hesapla
bedroom_avgs = df.groupby(['property_type', 'room_type', 'accommodates'])['bedrooms'].mean()

# Eksik 'bedrooms' değerlerini doldur
df['bedrooms'] = df.apply(lambda row: fill_bedrooms(row, bedroom_avgs), axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.shape

# Eksik verileri doldurma için medyan değerleri hesaplama
median_values = {
    'host_activity_score': df['host_activity_score'].median(),
    'host_response_rate': df['host_response_rate'].median() / 100,
    'host_acceptance_rate': df['host_acceptance_rate'].median() / 100,
}

# Sayısal değerler için medyan ile doldurma
for column, median_val in median_values.items():
    df[column].fillna(median_val, inplace=True)

# Kategorik ve oransal değerler için mod ile doldurma
mode_values = {
    'host_is_superhost': df['host_is_superhost'].mode()[0],
    'host_response_time': df['host_response_time'].mode()[0],
    # Diğer kategorik ve oransal sütunlar için benzer şekilde...
}

for column, mode_val in mode_values.items():
    df[column].fillna(mode_val, inplace=True)

# 'bedrooms', 'beds' gibi sütunlar için benzer ilanların medyanlarını kullanarak doldurma
# Örneğin, 'bedrooms':
grouped_data = df.groupby(['property_type', 'room_type', 'accommodates'])
df['bedrooms'] = df.apply(
    lambda row: grouped_data['bedrooms'].median().loc[row['property_type'], row['room_type'], row['accommodates']]
    if pd.isna(row['bedrooms']) else row['bedrooms'], axis=1
)

df.head()



# Eksik değerleri uygun şekilde dolduralım
# Öncelikle 'bedrooms', 'beds' ve 'bathrooms_text' için medyan değerleri kullanalım
df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
df['beds'].fillna(df['beds'].median(), inplace=True)
df['bathrooms_text'].fillna('1 bath', inplace=True)  # En yaygın değer olan '1 bath' ile dolduralım

# Değerlendirme puanları için ortalama değerleri kullanalım
review_score_columns = [
    'review_scores_checkin', 'review_scores_value', 'review_scores_cleanliness',
    'review_scores_location', 'review_scores_accuracy', 'review_scores_communication'
]
for col in review_score_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# 'host_reliability_score' için ortalama ile dolduralım
df['host_reliability_score'].fillna(df['host_reliability_score'].mean(), inplace=True)

# Minimum ve maksimum gecelik sayıları için medyan değerleri kullanalım
night_columns = [
    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
    'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'
]
for col in night_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Eksik değerlerin doldurulup doldurulmadığını kontrol edelim
df.isnull().sum()

df.to_excel("/Users/mrpurtas/Desktop/df1.xlsx", index=True)
df.columns
df.property_type.value_counts()


# 'property_type' ve 'room_type' sütunlarından yeni özellikler türetelim

# Öncelikle, her iki sütunun benzersiz değerlerini inceleyelim
unique_property_types = df['property_type'].unique()
unique_room_types = df['room_type'].unique()

# Örnek olarak, bazı yaygın mülk tipleri ve oda türleri için ikili (binary) özellikler oluşturalım
common_property_types = ['Apartment', 'House', 'Condominium', 'Loft', 'Serviced apartment']
common_room_types = ['Entire home/apt', 'Private room', 'Shared room']

# Bu kategoriler için ikili özellikler oluşturalım
for prop_type in common_property_types:
    df[f'property_type_{prop_type}'] = df['property_type'].apply(lambda x: 1 if prop_type in x else 0)

for room_type in common_room_types:
    df[f'room_type_{room_type}'] = df['room_type'].apply(lambda x: 1 if room_type in x else 0)

# Oluşturulan yeni özelliklerin ilk 5 kaydını gösterelim
df[[f'property_type_{prop_type}' for prop_type in common_property_types] +
                      [f'room_type_{room_type}' for room_type in common_room_types]].head()

# "Uzun Süreli Uygunluk Skoru" ve "Fiyat Verimliliği Skoru" gibi yeni özellikler türetelim

# Uzun süreli uygunluk skoru: Daha uzun süre uygun olan ilanlar daha yüksek skor alır
# Bu skoru hesaplamak için, 365 günlük uygunluk süresini baz alıp diğer uygunluk sürelerini orantılayacağız
df['long_term_availability_score'] = (
    df['availability_30'] / 30 +
    df['availability_60'] / 60 +
    df['availability_90'] / 90 +
    df['availability_365'] / 365
) / 4  # 4 farklı süre için ortalama alıyoruz

# Fiyat verimliliği skoru: Daha fazla kişiyi ağırlayan ve daha fazla yatak/banyo sunan ilanlar için fiyat verimliliği
# Fiyatı sayısal bir değere dönüştürmemiz gerekiyor
df['price_numeric'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['price_efficiency_score'] = df['price_numeric'] / (
    df['accommodates'] +
    df['beds'] +
    df['weighted_bathrooms']
)

# Yeni özelliklerin ilk 5 kaydını gösterelim
df[['long_term_availability_score', 'price_efficiency_score']].head()
############################################################################################################################################################################
import re

def extract_bathroom_details(text):
    match = re.search(r'(\d+\.?\d*)', text)
    number_of_bathrooms = float(match.group(1)) if match else None

    # Banyo tipini belirle ve ağırlık ata
    text_lower = text.lower()
    if 'shared' in text_lower:
        bathroom_type = 'shared'
        weight = 0.7  # Örnek ağırlık değeri, paylaşılan banyo için
    elif 'private' in text_lower:
        bathroom_type = 'private'
        weight = 1.5  # Örnek ağırlık değeri, özel banyo için
    else:
        bathroom_type = 'normal'
        weight = 1  # Normal banyo için varsayılan ağırlık

    weighted_bathrooms = number_of_bathrooms * weight if number_of_bathrooms is not None else None

    return weighted_bathrooms


df['weighted_bathrooms'] = df['bathrooms_text'].apply(extract_bathroom_details)
df.weighted_bathrooms.head()

df['popularity_score'] = (
    df['number_of_reviews'] + df['reviews_per_month']
)

df.columns

# Tarihleri datetime objesine dönüştürme ve bugünkü tarihle farkı hesaplama
df['host_since'] = pd.to_datetime(df['host_since'])
df['first_review'] = pd.to_datetime(df['first_review'])
df['last_review'] = pd.to_datetime(df['last_review'])

from datetime import datetime

# Mevcut tarihi alalım
current_date = pd.to_datetime('today')

# Ev Sahibi Deneyimi Süresi
df['host_experience_days'] = (current_date - df['host_since']).dt.days

# İlanın Piyasada Kalma Süresi
df['listing_duration_days'] = (current_date - df['first_review']).dt.days

# Son Aktivite Süresi
df['days_since_last_review'] = (current_date - df['last_review']).dt.days

# Oluşturulan yeni özelliklerin ilk 5 kaydını gösterelim
df[['host_experience_days', 'listing_duration_days', 'days_since_last_review']].head()



missing_values = df.isnull().sum()
missing_percent = (missing_values / df.shape[0]) * 100

# Eksik değer tablosunu oluşturalım
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percent': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

weighted_bathrooms_median = df['weighted_bathrooms'].median()
price_efficiency_score_median = df['price_efficiency_score'].median()

df['weighted_bathrooms'].fillna(weighted_bathrooms_median, inplace=True)
df['price_efficiency_score'].fillna(price_efficiency_score_median, inplace=True)

df.to_excel("/Users/mrpurtas/Desktop/dfson.xlsx", index=True)

df.columns

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

plt.figure(figsize=(20, 15))
for i, feature in enumerate(num_cols):
    plt.subplot(len(num_cols) // 3 + 1, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()

df.head()

for feature in num_cols:
    plt.figure(figsize=(10, 6))  # Her bir grafik için boyut ayarı
    sns.histplot(df[feature], kde=True)  # Histogram ve KDE
    plt.title(feature)  # Grafik başlığı
    plt.show()


# Veri setindeki sayısal değişkenlerin dağılımlarına bakarak logaritmik dönüşüm uygulayalım.
# Sıfır değerleri olan sütunlar için log1p kullanacağız ki log(0) tanımsızdır.

# log1p dönüşümü uygulanacak sütunlar
log_transform_columns = ['host_listings_count', 'number_of_reviews', 'reviews_per_month', 'price_numeric']

# Sütunlar üzerinde dönüşüm uygulayalım ve yeni sütunlar oluşturalım
for col in log_transform_columns:
    df[f'log_{col}'] = df[col].apply(lambda x: np.log1p(x))

# Oluşturulan yeni logaritmik sütunların ilk 5 kaydını gösterelim
df[[f'log_{col}' for col in log_transform_columns]].head()

# Dağılımları görselleştirelim
plt.figure(figsize=(20, 5))
for i, col in enumerate(log_transform_columns):
    plt.subplot(1, len(log_transform_columns), i + 1)
    sns.histplot(df[f'log_{col}'], kde=True)
    plt.title(f'log_{col}')
plt.tight_layout()
plt.show()

zero_values_in_listing_duration = (df['listing_duration_days'] == 0).any()
df['log_listing_duration_days'] = np.log(df['listing_duration_days'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_listing_duration_days'], kde=True)
plt.title('Log Transformed "listing_duration_days" Distribution')
plt.xlabel('log_listing_duration_days')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['host_experience_days'] == 0).any()
df['log_host_experience_days'] = np.log(df['host_experience_days'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['host_experience_days'], kde=True)
plt.title('Log Transformed "host_experience_days" Distribution')
plt.xlabel('log_listing_duration_days')
plt.ylabel('Frequency')
plt.show()

# 'popularity_score' sütununa logaritmik dönüşüm uygulayalım
df['log_popularity_score'] = np.log1p(df['popularity_score'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_popularity_score'], kde=True)
plt.title('Log Transformed "popularity_score" Distribution')
plt.xlabel('log_popularity_score')
plt.ylabel('Frequency')
plt.show()

df['log_price_efficiency_score'] = np.log1p(df['price_efficiency_score'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_price_efficiency_score'], kde=True)
plt.title('Log Transformed "log_price_efficiency_score" Distribution')
plt.xlabel('log_price_efficiency_score')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['weighted_bathrooms'] == 0).any()
df['log_weighted_bathrooms'] = np.log1p(df['weighted_bathrooms'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_weighted_bathrooms'], kde=True)
plt.title('Log Transformed "log_weighted_bathrooms" Distribution')
plt.xlabel('log_weighted_bathrooms')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['price_numeric'] == 0).any()
df['log_price_numeric'] = np.log(df['price_numeric'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_price_numeric'], kde=True)
plt.title('Log Transformed "log_price_numeric" Distribution')
plt.xlabel('log_price_numeric')
plt.ylabel('Frequency')
plt.show()
########################################################################################################################
#long_term_availability_score
# review_scores_rating
# last_reviewaykırılık olmayabılır ıncele
########################################################################################################################
# 'host_experience_days' sütununa logaritmik dönüşüm uygulayalım
df['log_host_experience_days'] = np.log(df['host_experience_days'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_host_experience_days'], kde=True)
plt.title('Log Transformed "log_host_experience_days" Distribution')
plt.xlabel('log_host_experience_days')
plt.ylabel('Frequency')
plt.show()

# 'number_of_reviews_l30d' sütununa logaritmik dönüşüm uygulayalım
df['log_number_of_reviews_l30d'] = np.log1p(df['number_of_reviews_l30d'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_number_of_reviews_l30d'], kde=True)
plt.title('Log Transformed "number_of_reviews_l30d" Distribution')
plt.xlabel('log_number_of_reviews_l30d')
plt.ylabel('Frequency')
plt.show()


# 'number_of_reviews_ltm' sütununa logaritmik dönüşüm uygulayalım
df['log_number_of_reviews_ltm'] = np.log1p(df['number_of_reviews_ltm'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_number_of_reviews_ltm'], kde=True)
plt.title('Log Transformed "number_of_reviews_ltm" Distribution')
plt.xlabel('log_number_of_reviews_ltm')
plt.ylabel('Frequency')
plt.show()


zero_values_in_listing_duration = (df['maximum_nights_avg_ntm'] == 0).any()
df['log_maximum_nights_avg_ntm'] = np.log(df['maximum_nights_avg_ntm'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_maximum_nights_avg_ntm'], kde=True)
plt.title('Log Transformed "log_maximum_nights_avg_ntm" Distribution')
plt.xlabel('log_maximum_nights_avg_ntm')
plt.ylabel('Frequency')
plt.show()


zero_values_in_listing_duration = (df['maximum_maximum_nights'] == 0).any()
df['log_maximum_maximum_nights'] = np.log(df['maximum_maximum_nights'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_maximum_maximum_nights'], kde=True)
plt.title('Log Transformed "log_maximum_maximum_nights" Distribution')
plt.xlabel('log_maximum_maximum_nights')
plt.ylabel('Frequency')
plt.show()


zero_values_in_listing_duration = (df['minimum_minimum_nights'] == 0).any()
df['log_minimum_minimum_nights'] = np.log(df['minimum_minimum_nights'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_minimum_minimum_nights'], kde=True)
plt.title('Log Transformed "log_minimum_minimum_nights" Distribution')
plt.xlabel('log_minimum_minimum_nights')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['maximum_nights'] == 0).any()
df['log_maximum_nights'] = np.log(df['maximum_nights'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_maximum_nights'], kde=True)
plt.title('Log Transformed "log_maximum_nights" Distribution')
plt.xlabel('log_maximum_nights')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['minimum_nights'] == 0).any()
df['log_minimum_nights'] = np.log(df['minimum_nights'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_minimum_nights'], kde=True)
plt.title('Log Transformed "log_minimum_nights" Distribution')
plt.xlabel('log_minimum_nights')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['minimum_maximum_nights'] == 0).any()
df['log_minimum_maximum_nights'] = np.log(df['minimum_maximum_nights'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_minimum_maximum_nights'], kde=True)
plt.title('Log Transformed "log_minimum_maximum_nights" Distribution')
plt.xlabel('log_minimum_maximum_nights')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['beds'] == 0).any()
df['log_beds'] = np.log(df['beds'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_beds'], kde=True)
plt.title('Log Transformed "log_beds" Distribution')
plt.xlabel('log_beds')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['bedrooms'] == 0).any()
df['log_bedrooms'] = np.log(df['bedrooms'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_bedrooms'], kde=True)
plt.title('Log Transformed "log_bedrooms" Distribution')
plt.xlabel('log_bedrooms')
plt.ylabel('Frequency')
plt.show()

zero_values_in_listing_duration = (df['host_total_listings_count'] == 0).any()
df['log_host_total_listings_count'] = np.log(df['host_total_listings_count'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_host_total_listings_count'], kde=True)
plt.title('Log Transformed "log_host_total_listings_count" Distribution')
plt.xlabel('log_host_total_listings_count')
plt.ylabel('Frequency')
plt.show()


df['log_host_listings_count'] = np.log(df['host_listings_count'])

# Dönüşüm sonrası dağılımı kontrol etmek için histogramını çizelim
sns.histplot(df['log_host_listings_count'], kde=True)
plt.title('Log Transformed "log_host_listings_count" Distribution')
plt.xlabel('log_host_listings_count')
plt.ylabel('Frequency')
plt.show()

df.to_excel("/Users/mrpurtas/Desktop/df2.xlsx", index=True)

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
dff.head()
dff.dtypes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Korelasyon matrisini hesaplama
corr_matrix = df.corr()
plt.figure(figsize=(24, 20))
sns.heatmap(corr_matrix, annot=False, fmt=".2f", square=True, cmap="coolwarm")
plt.title('Korelasyon Matrisi')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


def plot_correlation_matrix(df, figsize=(16, 12), cmap="coolwarm", annot=False, **kwargs):
    # Korelasyon matrisini hesaplama
    corr_matrix = df.corr()

    # Heatmap çizme
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=".2f",
        square=True,
        cmap=cmap,
        **kwargs,
    )

    # Başlık ve eksen etiketleri ekleme
    plt.title("Korelasyon Matrisi")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Korelasyon katsayılarının sınırlarını ayarlama
    vmin = -1
    vmax = 1
    plt.clim(vmin=vmin, vmax=vmax)

    # Çok küçük korelasyon katsayılarını gizle
    threshold = 0.1
    plt.axhline(threshold, color="k", linestyle="--")
    plt.axvline(threshold, color="k", linestyle="--")

    # Grafiği gösterme
    plt.show()



plot_correlation_matrix(df)


# Yüksek korelasyonlu özellikleri belirleme (örneğin, korelasyon katsayısı 0.7'den büyük olanlar)
high_corr_features = set()
threshold = 0.7

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

print("Yüksek Korelasyonlu Özellikler:", high_corr_features)


# Hedef değişkeninizi 'target' olarak varsayıyorum, değişken adınız farklı ise güncelleyin.
target_corr = df.corrwith(df['log_price_numeric'])
target_corr = target_corr.sort_values(ascending=False)

print("Hedef Değişken ile Korelasyonlar:\n", target_corr)

"""Hedef Değişken ile Korelasyonlar:
 log_price_numeric                              1.000000
log_price_efficiency_score                       0.796078
accommodates                                    0.520570
log_weighted_bathrooms                          0.408630
room_type_Entire home/apt                       0.401914
log_beds                                        0.389734
log_bedrooms                                    0.387705
beds                                            0.283241
weighted_bathrooms                              0.262477
price_efficiency_score                           0.257062
bedrooms                                        0.255460
price_numeric                                   0.243072
log_host_listings_count                         0.199689
log_host_total_listings_count                   0.198902
is_multi_host                                   0.142020
host_activity_score                             0.138757
maximum_maximum_nights                          0.137255
log_maximum_nights_avg_ntm                      0.133366
log_maximum_nights                              0.120747
log_minimum_maximum_nights                      0.120282
host_verification_count                          0.115113
host_acceptance_rate                            0.103878
calculated_host_listings_count_entire_homes     0.099865
latitude                                        0.098519
host_experience_score                           0.098242
host_duration                                   0.095650
calculated_host_listings_count                  0.092836
host_listings_count                             0.092305
host_experience_days                            0.088978
host_about_length                               0.087385
long_term_availability_score                    0.087153
maximum_nights_avg_ntm                          0.087146
log_host_experience_days                        0.087078
host_total_listings_count                       0.084005
minimum_maximum_nights                          0.082599
availability_90                                 0.080924
availability_30                                 0.079368
availability_60                                 0.077805
number_of_reviews                               0.072897
popularity_score                                0.072665
average_monthly_reviews                         0.071639
availability_365                                0.071356
log_listing_duration_days                       0.070426
host_is_superhost                               0.070144
host_reliability_score                          0.068439
maximum_nights                                  0.066551
listing_duration_days                           0.065605
log_popularity_score                            0.059517
review_scores_location                          0.059499
number_of_reviews_ltm                           0.054772
minimum_nights                                  0.054735
review_scores_cleanliness                       0.049981
activity_duration                               0.049485
log_minimum_nights                              0.041949
days_since_last_review                          0.041815
host_response_rate                              0.032474
weighted_total_review_score                     0.029143
review_scores_checkin                           0.027658
longitude                                       0.027418
reviews_per_month                               0.027081
total_review_score                              0.024464
maximum_minimum_nights                          0.022907
review_scores_rating                            0.022051
log_number_of_reviews_ltm                       0.020996
review_scores_accuracy                          0.020420
minimum_nights_avg_ntm                          0.019886
host_has_profile_pic                            0.017237
minimum_minimum_nights                          0.014873
log_minimum_minimum_nights                      0.014775
review_scores_communication                     0.014399
host_identity_verified                          0.005283
log_number_of_reviews_l30d                     -0.000076
number_of_reviews_l30d                         -0.000174
review_scores_value                            -0.007666
id                                             -0.060113
calculated_host_listings_count_shared_rooms    -0.084490
calculated_host_listings_count_private_rooms   -0.087286
host_id                                        -0.088602
room_type_Shared room                          -0.102002
host_response_time                             -0.112230
room_type_Private room                         -0.397964
scrape_id                                            NaN
property_type_Apartment                              NaN
property_type_House                                  NaN
property_type_Condominium                            NaN
property_type_Loft                                   NaN
property_type_Serviced apartment                     NaN
log_maximum_maximum_nights                           NaN
dtype: float64
"""


dff = df.copy()
import pandas as pd

# 'host_since', 'first_review', 'last_review' sütunlarını datetime türüne dönüştür
dff['host_since'] = pd.to_datetime(dff['host_since'])
dff['first_review'] = pd.to_datetime(dff['first_review'])
dff['last_review'] = pd.to_datetime(dff['last_review'])




dff.columns.tolist()

dff.nunique()

dff['host_response_rate'] = dff['host_response_rate'].astype('float')
dff['host_acceptance_rate'] = dff['host_acceptance_rate'].astype("float")
dff['host_response_time'] = dff['host_response_time'].astype('float')

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

binary_cols = [col for col in df.columns if dff[col].dtypes == "O" and dff[col].nunique() == 2 and col not in ["log_price_numeric"]]

dff['booking_flexibility'] = dff['availability_365'] / dff['minimum_nights']

label_encoder = LabelEncoder()

# 'instant_bookable' sütununu dönüştür
dff['instant_bookable_encoded'] = label_encoder.fit_transform(dff['instant_bookable'])
dff['is_multi_host'] = label_encoder.fit_transform(dff['is_multi_host'])
dff['has_availability'] = label_encoder.fit_transform(dff['has_availability'])



cat_cols, num_cols, cat_but_car = grab_col_names(dff)

# Mevcut num_cols listesine yeni sütunları eklemek
additional_cols = [
    'host_response_time',
    'host_is_superhost',
    'host_has_profile_pic',
    'host_identity_verified',
    'is_multi_host',
    'host_verification_count',
    'host_reliability_score',
    'property_type_Apartment',
    'property_type_House',
    'property_type_Condominium',
    'property_type_Loft',
    'property_type_Serviced apartment',
    'room_type_Entire home/apt',
    'room_type_Private room',
    'room_type_Shared room'
]

# Mevcut listeyle yeni sütunları birleştir
num_cols = num_cols + additional_cols

X = dff.drop(["id", "source", "listing_url", "scrape_id", "last_scraped", "name", "description", "neighborhood_overview",
              "picture_url", "host_id", "host_url", "host_name", "host_since", "host_location", "host_about",
              "host_thumbnail_url", "host_picture_url", "host_verifications", "host_listings_count", "host_total_listings_count",
              "property_type", "room_type", "bathrooms_text", "amenities", "price", "bedrooms", "beds", "minimum_nights",
              "maximum_nights", "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
              "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
              "first_review", "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
              "review_scores_location", "review_scores_value", "instant_bookable", "calculated_host_listings_count", "total_review_score", "price_numeric", "host_experience_days",
              "listing_duration_days", "popularity_score", "price_efficiency_score", "weighted_bathrooms", "log_price_numeric", "number_of_reviews_l30d", "number_of_reviews_ltm",
              "calendar_last_scraped", "last_review", "host_duration", "has_availability", "average_monthly_reviews", "log_maximum_nights_avg_ntm", "availability_90", "availability_60", "log_host_total_listings_count",
              "availability_30", "availability_365", "log_maximum_maximum_nights", "calculated_host_listings_count_entire_homes", "host_response_time", "log_minimum_nights", "log_number_of_reviews_ltm"], axis=1)
y = dff["log_price_numeric"]
dff.head()

X.head()
X.columns.tolist()







X.to_csv("/Users/mrpurtas/Desktop/dff_X.csv", index=True)




# Korelasyon matrisini hesaplayın
corr_matrix = X.corr()

plt.figure(figsize=(20, 20))  # Figür boyutunu artır
sns.set(font_scale=1.25)  # Etiket boyutunu artır
corr_matrix = X.corr()  # Korelasyon matrisini hesapla
sns.heatmap(corr_matrix, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title("Bağımsız Değişkenler Arasındaki Korelasyon", fontsize=20)
plt.xticks(rotation=90)  # X eksenindeki etiketleri 90 derece döndür
plt.yticks(rotation=0)  # Y eksenindeki etiketleri düz tut
plt.tight_layout()  # Çıktıyı düzgün hale getir
plt.show()





# Bağımlı değişkeninizi belirleyin
dependent_variable = 'log_price_numeric'

# Bağımlı değişken ile diğer tüm değişkenler arasındaki korelasyonları hesaplayın
correlations = dff.corr()[dependent_variable]

# Korelasyonları gösteren DataFrame oluşturun
correlation_df = pd.DataFrame(correlations).reset_index()
correlation_df.columns = ['Variable', 'Correlation_with_Dependent_Variable']

# Korelasyonları büyükten küçüğe sıralayın
correlation_df = correlation_df.sort_values(by='Correlation_with_Dependent_Variable', ascending=False)

# Sonuçları yazdırın
print(correlation_df)


from sklearn.preprocessing import StandardScaler

# Ölçeklendirilecek değişkenlerin listesi
scale_columns = ['host_response_rate', 'host_acceptance_rate', 'latitude', 'longitude', 'accommodates',
                 'number_of_reviews', 'reviews_per_month', 'weighted_total_review_score',
                 'days_since_last_review', 'activity_duration', 'host_about_length',
                 'host_verification_count', 'host_activity_score', 'host_reliability_score',
                 'host_experience_score', 'long_term_availability_score', 'log_listing_duration_days',
                 'log_host_experience_days', 'log_popularity_score', 'log_price_efficiency_score',
                 'log_weighted_bathrooms', 'log_number_of_reviews_l30d', 'log_minimum_minimum_nights',
                 'log_maximum_nights', 'log_minimum_maximum_nights', 'log_beds', 'log_bedrooms',
                 'log_host_listings_count']

# StandardScaler nesnesi oluşturma
scaler = StandardScaler()

# Seçilen değişkenleri ölçeklendir
dff[scale_columns] = scaler.fit_transform(dff[scale_columns])

#useful featureların sayısı kullanıldı
dff["amenities_count"] = dff["amenities"].str.split(",").apply(lambda x: len(x))

dff['amenities_count'] = scaler.fit_transform(dff[['amenities_count']])

df.to_csv("/Users/mrpurtas/Desktop/dfwithoutscale.csv", index=True)
dff.to_csv("/Users/mrpurtas/Desktop/dffwithscale.csv", index=True)

df = pd.read_csv('datasets/dfwithoutscale.csv', low_memory=False)
dff = pd.read_csv('datasets/dffwithscale.csv', low_memory=False)
############# DOĞAL DİL İŞLEME #############    # NLP
# df ve reviews isimli iki DataFrame varsayalım ve her ikisinin de 'id' adında bir sütunu olsun.
# 'id' sütunu üzerinden df'i reviews ile birleştirelim.
reviews = pd.read_csv('datasets/reviews.csv')

merged_df = pd.merge(df, reviews[['listing_id', 'comments']], left_on='id', right_on='listing_id', how='left')
merged_dff = pd.merge(dff, reviews[['listing_id', 'comments']], left_on='id', right_on='listing_id', how='left')

# Sonuçları kontrol edin
print(merged_df.head())

reviews.head()# Bu komut, df'teki her satır için 'id'ye göre reviews'teki ilgili satırları arar ve bunları df'e ekler.
merged_dff.isna().sum()# Bu komut, df'teki her satır için 'id'ye göre reviews'teki ilgili satırları arar ve bunları df'e ekler.
merged_dff.dropna(subset=['comments'], inplace=True)
merged_dff.head()

df.shape

####GORSELLESTIRMELER - HARITA
plt.figure(figsize=(12, 8))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_cleansed', data=df, alpha=0.6, edgecolor='k', palette='viridis')
plt.title('Mahallelere Göre Konum Dağılımı')
plt.xlabel('Boylam (Longitude)')
plt.ylabel('Enlem (Latitude)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # Eğer legend çok büyükse veya grafiği kaplıyorsa, dışarı taşıyabilirsiniz.
plt.show()



############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################


#dff.to_csv("/Users/mrpurtas/Desktop/dff_enson.csv", index=True)





X = dff.drop(["Unnamed: 0", "id", "source", "listing_url", "scrape_id", "last_scraped", "name", "description", "neighborhood_overview",
              "picture_url", "host_id", "host_url", "host_name", "host_since", "host_location", "host_about",
              "host_thumbnail_url", "host_picture_url", "host_verifications", "host_listings_count", "host_total_listings_count",
              "property_type", "room_type", "bathrooms_text", "amenities", "price", "bedrooms", "beds", "minimum_nights",
              "maximum_nights", "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
              "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
              "first_review", "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
              "review_scores_location", "review_scores_value", "instant_bookable", "calculated_host_listings_count", "total_review_score", "price_numeric", "host_experience_days",
              "listing_duration_days", "popularity_score", "price_efficiency_score", "weighted_bathrooms", "log_price_numeric", "number_of_reviews_l30d", "number_of_reviews_ltm",
              "calendar_last_scraped", "last_review", "host_duration", "has_availability", "average_monthly_reviews", "log_maximum_nights_avg_ntm", "availability_90", "availability_60", "log_host_total_listings_count",
              "availability_30", "log_price_efficiency_score", "availability_365", "log_maximum_maximum_nights", "calculated_host_listings_count_entire_homes", "host_response_time", "log_minimum_nights", "log_number_of_reviews_ltm", "host_response_rate"], axis=1)
y = dff["log_price_numeric"]
X.columns.tolist()
dff.head()
def base_models_regression_with_cv(X, y, cv=5):
    print("Base Models with Cross-Validation for Regression....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressors = [
        ('Linear Regression', LinearRegression()),
        ('Ridge', Ridge(alpha=1.0)),
        ('Lasso', Lasso(alpha=0.1)),
        ('Decision Tree Regressor', DecisionTreeRegressor()),
        ('Random Forest Regressor', RandomForestRegressor(n_estimators=100)),
        ('Gradient Boosting Regressor', GradientBoostingRegressor()),
        ('XGBoost Regressor', XGBRegressor()),
        ('LightGBM Regressor', LGBMRegressor())
    ]

    for name, regressor in regressors:
        scores = cross_val_score(regressor, X, y, cv=cv, scoring='r2')
        mse_scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_absolute_error')

        print(f"{name} Regressor:")
        print(f"Mean R^2 Score: {np.mean(scores)} (std: {np.std(scores)})")
        print(f"Mean Negative Mean Squared Error: {np.mean(mse_scores)} (std: {np.std(mse_scores)})")
        print(f"Mean Negative Mean Absolute Error: {np.mean(mae_scores)} (std: {np.std(mae_scores)})\n")

# Fonksiyonu kullanma örneği
base_models_cv = base_models_regression_with_cv(X, y, cv=5)

dff["log_price_numeric"]




"""Base Models with Cross-Validation for Regression....
Linear Regression Regressor:
Mean R^2 Score: 0.46655479208859135 (std: 0.06816048795313805)
Mean Negative Mean Squared Error: -0.29544636300245775 (std: 0.06578582247050764)
Mean Negative Mean Absolute Error: -0.38953573174522216 (std: 0.03956084181835164)

Ridge Regressor:
Mean R^2 Score: 0.4661823349242994 (std: 0.06847179975981756)
Mean Negative Mean Squared Error: -0.29564341533870014 (std: 0.06588879648369449)
Mean Negative Mean Absolute Error: -0.389645538123657 (std: 0.03961672799079912)

Lasso Regressor:
Mean R^2 Score: 0.2783292201435065 (std: 0.03095134715111483)
Mean Negative Mean Squared Error: -0.39792109431347683 (std: 0.06485146869044908)
Mean Negative Mean Absolute Error: -0.4663200805582388 (std: 0.03625041589470744)

Decision Tree Regressor Regressor:
Mean R^2 Score: 0.011857852252732014 (std: 0.1702181668491192)
Mean Negative Mean Squared Error: -0.5501045726322136 (std: 0.1418762451841001)
Mean Negative Mean Absolute Error: -0.5078147123955021 (std: 0.04959332551082779)

Random Forest Regressor Regressor:
Mean R^2 Score: 0.5501282044459946 (std: 0.04340084405221668)
Mean Negative Mean Squared Error: -0.24846817551384656 (std: 0.05024505917348719)
Mean Negative Mean Absolute Error: -0.3438512232792111 (std: 0.027311485594995293)

Gradient Boosting Regressor Regressor:
Mean R^2 Score: 0.5447325297490074 (std: 0.03842389773103291)
Mean Negative Mean Squared Error: -0.2512147473855092 (std: 0.0468527398601189)
Mean Negative Mean Absolute Error: -0.35316426120662825 (std: 0.026682626465516106)

XGBoost Regressor Regressor:
Mean R^2 Score: 0.5491368732491528 (std: 0.08128297125489993)
Mean Negative Mean Squared Error: -0.25018719645555754 (std: 0.0662555984258143)
Mean Negative Mean Absolute Error: -0.3411407494684206 (std: 0.0400739010853254)

LightGBM Regressor Regressor:
Mean R^2 Score: 0.5846428518561432 (std: 0.04245358991883646)
Mean Negative Mean Squared Error: -0.22974499055571945 (std: 0.046245732710085925)
Mean Negative Mean Absolute Error: -0.33048396633126725 (std: 0.027696792541411566)
"""


# Veri setini eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM modelini oluştur
lgbm = lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.1, reg_alpha=0.2, reg_lambda=0.1)

# Eğitim setinde çapraz doğrulama yap
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(lgbm, X_train, y_train, cv=kf, scoring='r2')
cv_mse_scores = cross_val_score(lgbm, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

print("Eğitim Seti - Ortalama R^2 Skoru:", np.mean(cv_r2_scores))
print("Eğitim Seti - Ortalama MSE Skoru:", -np.mean(cv_mse_scores))

# Modeli tüm eğitim seti üzerinde eğit
lgbm.fit(X_train, y_train)

# Test seti üzerinde tahmin yap ve değerlendir
y_pred = lgbm.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Test Seti - R^2 Skoru:", r2)
print("Test Seti - MSE Skoru:", mse)
#Test Seti - MSE Skoru: 0.22564017990378618
#Test Seti - R^2 Skoru: 0.6281685249916581


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm, X)

from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
import numpy as np


from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
import lightgbm as lgb
import numpy as np

# Veri setini eğitim ve test setlerine ayır (Varsayalım ki bu daha önce yapılmış)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM parametre aralıkları
lightgbm_params = {
    "num_leaves": range(20, 200, 30),
    "learning_rate": np.arange(0.01, 0.3, 0.05),
    "n_estimators": range(50, 300, 50),
    "subsample": np.arange(0.5, 1.05, 0.1),
    "colsample_bytree": np.arange(0.5, 1.05, 0.1)
}

# LightGBM modelini oluştur
lgbm = lgb.LGBMRegressor()

# Randomized Grid Search modelini oluştur
random_search = RandomizedSearchCV(lgbm, param_distributions=lightgbm_params, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)

# Randomized Grid Search'i çalıştır
random_search.fit(X_train, y_train)

# En iyi parametrelerle yeni bir LightGBM modeli oluştur
best_params = random_search.best_params_
best_lgbm = lgb.LGBMRegressor(**best_params)

# KFold nesnesini oluştur
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# R^2 ve MSE için çapraz doğrulama skorlarını hesapla
cv_r2_scores = cross_val_score(best_lgbm, X, y, cv=kf, scoring='r2')
cv_mse_scores = -cross_val_score(best_lgbm, X, y, cv=kf, scoring='neg_mean_squared_error')

# Ortalamaları yazdır
print("Çapraz Doğrulama - Ortalama R^2 Skoru:", np.mean(cv_r2_scores))
print("Çapraz Doğrulama - Ortalama MSE Skoru:", np.mean(cv_mse_scores))
print("Çapraz Doğrulama - R^2 Skorlarının Standart Sapması:", np.std(cv_r2_scores))
print("Çapraz Doğrulama - MSE Skorlarının Standart Sapması:", np.std(cv_mse_scores))





