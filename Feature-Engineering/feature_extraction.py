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

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()

# notnull dan sonra True-False dönüyor onu int yaparak 1 veya 0 olmasını sağladık
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

# kabini dolu olanların hayatta kalma oranı, NAN olan kabinlere göre daha fazla
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


from statsmodels.stats.proportion import proportions_ztest

# kabin numarası olan ve hayatta kalan kaç kişi
# kabin numarası olmayan ve hayatta kalan kaç kişi
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
# H0 Reddedildi, ikisi arasında fark var
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 0 dan büyük ise yalnız değil,
# kişinin tek olması durumuna göre hayatta kalma çabasını gelişrtirmiş olabilir? bilemiyoruz
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

# H0 Reddedildi, ikisi arasında fark var
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count
###################
# harf sayımı
df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################
# kelime sayımı, boşlklara göre split, sonra len-boyutuna bak
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################
# isminde Dr(Doktor) ünvanı olanları al

# önce split et sonra bunlar da gez,
# eğer o gexdiğin her bir kelimenin başlangıcında Dr varsa al
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# Dr lanların hayatta kalma oranı daha yüksek
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

###################
# Regex ile Değişken Türetmek
###################

df.head()

# Name sütununa baktığımızda boşlukla başlayıp . ile bitmiş,
# aralarına da büyük veya küçük harfler oluşacak şekilde göreceğin harfleri yakala
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# isimlendirmelere göre Survived ve Age değişken kırılımı
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("Feature-Engineering/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
# yıl farkını alıp bunu 12 ile çarpıp ay cinsine çeviriyoruz
# sonra ay farkını alıp topluyoruz
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date

#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
# iki değişkenin birbiri ile etkileşimi

df = load()
df.head()

# yaşı büyük veya küçük olanların, yolculuk sınıflarına göre refah durumlarıyla ilgili durum ortaya çıkarmak
# mesela yaşı küçük 1. sınıf refah seviyesi yüksek mi
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

# aile bireylerinin sayısıyla aile yapısı
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

# olgun kadınların, hayatta kalma oranları daha yüksek geldi mesela
df.groupby("NEW_SEX_CAT")["Survived"].mean()
