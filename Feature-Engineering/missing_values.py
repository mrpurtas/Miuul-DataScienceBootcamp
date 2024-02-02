#############################################
# Missing Values (Eksik Değerler)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################################
# Eksik Değerlerin Yakalanması
#############################################

def load_application_train():
    data = pd.read_csv("Feature-Engineering/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("Feature-Engineering/titanic.csv")
    return data

df = load()
df.head()

df = load()
df.head()

# eksik gozlem var mı yok mu
df.isnull().values.any()

# degiskenlerdei eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# yuzdelik degerlerini gorme, yuzde kaci eksik gibi
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# sadece eksik degerlere sahip degiskenler
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

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

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

missing_values_table(df, True)


#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)  # satırların ortalamasını alacağımız için axis=0

dff = df.apply(lambda x: x.fillna(x.mean())if x.dtype != "O" else x, axis=0)  # object olmayan değerleri doldur

dff.isnull().sum().sort_values(ascending=False)

# kategorik değişkenleri doldurmak için en iyi yöntem modunu almaktır
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

# tipi object olan ve eşsiz değer sayısı 10'dan(yoruma açık) küçük olan,
# bir kategorik değişken ise bunu modunu değişkene ata
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

##############################################
# Kategorik Değişken Kırılımında Değer Atama
##############################################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

# kadınlardaki eksiklikleri, kırılımdaki kadınların ortalaması ile doldurduk
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

# erkekler için de aynısını yapıyoruz
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()


#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# get_dummies = sadece kategorik değişkenlere dönüşüm uygulamaktadır,
# onun için cat_cols'u çğırdık
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)  # en yakın 5 komşu
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)  # en yakın 5 komşunun dolu olan gözlemlerinin ortalamasını boş olan değerlere atar
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)  # geri alıyoruz, önceki halleriylekıyas yapamıyoruz çünkü

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]  # veri setimzideki eksik olan değerlerin yerine tahmin edilmiş halini atamış olduk
df.loc[df["Age"].isnull()]



#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#################################################################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
#################################################################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)  # seçtiğin ilgili değişken de eksiklik varsa 1, yoksa 0 yaz

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns  # tüm satırları getir ama içerisinde "_NA_" içeren sütunları seç getir

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

# "Survived" değişkeni ile bu eksikliğe sahip olan değişkenleri bi karşılaştır
# mesela "Age" değişkeninde dolu olan senaryonun hayatta kalma oranı,
# yine "Age" değişkeninde dolu olmayan senaryonun hayatta kalma oranı gibi
# NA olanlar 1, NA olmayan 0
missing_vs_target(df, "Survived", na_cols)

###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)

# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()

# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)

