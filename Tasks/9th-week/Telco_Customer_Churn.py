import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
matplotlib.use("Qt5Agg")
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skompiler import skompile
import graphviz

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv('datasets/machine_learning/Telco-Customer-Churn.csv')

df.head()

#genel resime bakalım:

def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, quan=True)


#Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols

num_cols

cat_but_car

df.value_counts()

#Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

df["TotalCharges"].dtypes

df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df["TotalCharges"].dtypes

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = df.drop(['customerID'], axis = 1)


#Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

# Kategorik Değişkenlerin Analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title('Kategorilerin Dağılımı')
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)

df.head()


# Numerik Değişkenlerin Analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

#Adım 4: Hedef değişken analizi yapınız.(Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

#Kategorik değişkenlerin hedef değişkene göre analizi


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Numerik değişkenlerin hedef değişkene göre analizi

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# Numerik değişkenlerin hedef değişkene göre analizi - Farklı yol

def mean_median_analyses_with_target(dataframe, col_name, target):
    print(pd.DataFrame({col_name + "_mean": dataframe.groupby(target)[col_name].mean(),
                        col_name + "_median": dataframe.groupby(target)[col_name].median()}))

    print("##########################################################")

for col in num_cols:
    mean_median_analyses_with_target(df, col, "Churn")



#########################################################################################
df("TotalCharges").describe().T

plt.figure(figsize=(8, 5))
sns.histplot(df['TotalCharges'], kde=True)
plt.title('Distribution of Total Charges')
plt.xlabel('Total Charges')
plt.ylabel('Frequency')
plt.show()

"""total_charges'a mean>median ve gorselleştirerek baktıgımızda saga carpık bır değişkendır aykırı 
 değerler sağda toplanmış denebilr"""

"""numerıc kolonlarda sıfırda bi yıgılma var sebebını bılmıyorum :)"""
#########################################################################################

#Adım 5: Aykırı gözlem var mı inceleyiniz.

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

#aykırı değer yok

#Adım 6: Eksik gözlem var mı inceleyiniz.


# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()



na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

#Görev 2 : Feature Engineering

#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

#değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir bakalım?

def maybe_missing(dataframe, col_name):
    observations = dataframe[dataframe[col_name] == 0].shape[0]
    return observations
df.head()

for col in num_cols:
    print(col, maybe_missing(df, col))
# 11 tane tenure değeri 11 gorunuyor bu musterının henuz sısteme kayıt oldugu anlamına gelebılır sorun yok

def missing_values_table(dataframe, na_name=False):
    """missing_values_table fonksiyonu, bir pandas DataFrame'i içerisinde eksik (NaN ya da None olarak temsil edilen)
    değerleri analiz eder ve bu eksik değerlerin her sütun için sayısını ve yüzdesini bir tablo olarak sunar."""
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df, na_name=True)

def missing_vs_target(dataframe, target, na_columns):
    """Bu fonksiyonun amacı, eksik verilerin hedef değişken üzerindeki potansiyel etkisini anlamaktır.
     Örneğin, eksik verilere sahip gözlemlerin hedef değişkenin ortalama değerinde bir farklılık gösterip
     göstermediğini analiz etmek için kullanılabilir"""
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Churn", na_cols)

"""TotalCharges sütununda eksik veri içeren gözlemlerin hiçbiri churn olmamış. Bu, eğer 
TotalCharges müşterinin toplamda ödediği ücreti ifade ediyorsa, eksik verilerin yeni 
müşterilerden (henüz ücret ödememiş olabilirler) veya kayıt hatasından kaynaklanmış 
olabileceğini düşündürebilir."""

# eksik değerlerin doldurulması

"""median ile doldurdum cunku "Total_Charges" değişkeninin veri dağılımı sağa carpık bi şekilde, yani mean yerıne
medıan kullanmak daha saglıklı olacaktır"""
for col in na_cols:
    df.loc[df[col].isnull(), col] = df[col].median()
df.head()

df.isnull().sum()

#Adım 2: Yeni değişkenler oluşturunuz.



df["tenure"]

df.nunique()
df["tenure"].describe().quantile(0, 0.)

df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 24, 48, 72],
    labels=["0-2Yıl", "2-4Yıl", "4-6Yıl"])

#musterı yasam boyu değeri

gross_margin = 0.5
df['CLV'] = (df['MonthlyCharges'] * gross_margin) * df['tenure']

müşterinin ortalama aylık harcamasının
df['AverageMonthlySpend'] = df['TotalCharges'] / (df['MonthlyCharges'])

df["Cat_MonthlyCharges"] = pd.cut(df["MonthlyCharges"], bins=[df.MonthlyCharges.min(), 40, 70, df.MonthlyCharges.max()],
                                  labels=["Lower", "Middle", "High"], right=False)





# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()
df.head()


# One Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ohe_cols)
df.head()


# Standardization for Numeric variables

for col in num_cols:
    print(col, check_outlier(df, col))

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()

df.drop('customerID', axis=1, inplace=True)

## Modelling

# Establishing models with classification algorithms

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

random_user = X.sample(1)  # rastgele bir kullanıcı oluşturuyoruz

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# Logistic Regression
log_model = LogisticRegression(max_iter=1000).fit(X, y)

# cross validation
log_cv_results = cross_validate(log_model,
                                X, y,  # bağımlı ve bağımsız değişkenler
                                cv=5,  # dördüyle model kur, biriyle test et
                                scoring=["accuracy", "f1", "roc_auc"])  # istediğimiz metrikler

log_test = log_cv_results['test_accuracy'].mean()
# 0.8059382470763069
log_f1 = log_cv_results['test_f1'].mean()
# 0.5911821068005934
log_auc = log_cv_results['test_roc_auc'].mean()
# 0.8463000059038415
log_model.predict(random_user)


# RandomForestClassifier
rf_model = RandomForestClassifier().fit(X, y)

# cross validation
rf_cv_results = cross_validate(rf_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])

rf_test = rf_cv_results['test_accuracy'].mean()
# 0.7910226666989727
rf_f1 = rf_cv_results['test_f1'].mean()
# 0.5533446100583161
rf_auc = rf_cv_results['test_roc_auc'].mean()
# 0.8244377769644089
rf_model.predict(random_user)


# GBM
gbm_model = GradientBoostingClassifier().fit(X, y)

# cross validation
gbm_cv_results = cross_validate(gbm_model, X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

gbm_test = gbm_cv_results['test_accuracy'].mean()
# 0.8053167408234181
gbm_f1 = gbm_cv_results['test_f1'].mean()
# 0.5916996422107582
gbm_auc = gbm_cv_results['test_roc_auc'].mean()
# 0.84598827585678


# LightGBM
lgbm_model = LGBMClassifier().fit(X, y)

# cross validation
lgbm_cv_results = cross_validate(lgbm_model,
                                 X, y,
                                 cv=5,
                                 scoring=["accuracy", "f1", "roc_auc"])

lgbm_test = lgbm_cv_results['test_accuracy'].mean()
# 0.7950782563508408
lgbm_f1 = lgbm_cv_results['test_f1'].mean()
# 0.5756609171367744
lgbm_auc = lgbm_cv_results['test_roc_auc'].mean()
# 0.8350053407214943
lgbm_model.predict(random_user)


# XGBoost
xgboost_model = XGBClassifier(use_label_encoder=False)

# cross validation
xg_cv_results = cross_validate(xgboost_model,
                               X, y,
                               cv=5,
                               scoring=["accuracy", "f1", "roc_auc"])

xg_test = xg_cv_results['test_accuracy'].mean()
# 0.7839858824147905
xg_f1 = xg_cv_results['test_f1'].mean()
# 0.5597840395783735
xg_auc = xg_cv_results['test_roc_auc'].mean()
# 0.8256516811522634


# K-NN
knn_model = KNeighborsClassifier().fit(X, y)

# cross validation
knn_cv_results = cross_validate(knn_model,
                                X, y,
                                cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

knn_test = knn_cv_results['test_accuracy'].mean()
# 0.7519915156992926
knn_f1 = knn_cv_results['test_f1'].mean()
# 0.4495756497728191
knn_auc = knn_cv_results['test_roc_auc'].mean()
# 0.7040636259728742
knn_model.predict(random_user)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)