
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('datasets/feature_engineering/diabetes.csv')
df.head()


##################################################################
# Gorev 1 : Keşifçi Veri Analizi
##################################################################
# Adım 1 : Genel Resmi Inceleyiniz


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

# Genel resmi incelerken dikkat etmemiz gerekenler veri setindeki değişkenlerimizin tipleri doğru atanmış mı? Boş gözlemler var mı? Sayısal değişkenlerimin veri setindeki dağılımları nasıl?"""

check_df(df, quan=False)

#Adım 2 :Numerik ve kategorik değişkenleri yakalayınız

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

#Adım 3 :Numerik ve kategorik değişkenlerin analizini yapınız.

# Kategorik Değişkenlerin Analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Yalnızca hedef değişkenimiz kategorik değişken olduğu için onu inceliyoruz. Modelleme aşamasına geldiğimizde sınıflandırma problemlerinde hedef değişkenimizin dağılımı bizim tahminlerimizde önemli rol oynayacak."""

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Numerik Değişkenlerin Analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles))

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
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_cols)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

# Numerik değişkenlerin hedef değişkene göre analizi

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Numerik değişkenlerin hedef değişkene göre analizi - Farklı yol

def mean_median_analyses_with_target(dataframe, col_name, target):
    print(pd.DataFrame({col_name + "_mean": dataframe.groupby(target)[col_name].mean(),
                        col_name + "_median": dataframe.groupby(target)[col_name].median()}))

    print("##########################################################")

for col in num_cols:
    mean_median_analyses_with_target(df, col, "Outcome")

# Adım 5 :Aykırı gözlem analizi yapınız.

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
    print(col, check_outlier(df, col))

# Adım 6:Eksik gözlem analizi yapınız.

df.isnull().sum()

# Adım 7: Korelasyon analizi yapınız.


f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdPu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

"""Glukoz ve Sonuç (Outcome): Glukoz ve Sonuç değişkenleri arasında 0.467 ile en yüksek pozitif korelasyona
 sahip, bu da daha yüksek glukoz seviyelerinin sonucun (muhtemelen diyabet olasılığının) artmasıyla ilişkili 
 olabileceğini gösterir.

Gebelik Sayısı (Pregnancies) ve Yaş: Gebelik sayısı ile Yaş arasında 0.544 ile orta derecede pozitif bir korelasyon
 bulunmakta, bu da yaşın artmasıyla gebelik sayısının artma eğilimi olduğunu gösterir.

İnsülin ve Cilt Kalınlığı (Skin Thickness): İnsülin ve Cilt Kalınlığı arasında 0.437 ile orta derecede pozitif bir 
korelasyon var, bu iki değişkenin birbiriyle ilişkili olabileceğini gösteriyor.

BMI ve Cilt Kalınlığı (Skin Thickness): BMI ve Cilt Kalınlığı arasındaki 0.393'lük korelasyon, bu iki ölçümün birbiriyle
 pozitif yönlü ilişkili olduğunu gösteriyor.

Yaş ve Sonuç (Outcome): Yaş ile Sonuç arasındaki 0.238'lik korelasyon, yaşın artmasıyla sonucun olumlu yönde 
etkilenebileceğini ancak bu ilişkinin glukoz kadar güçlü olmadığını gösterir.

Diğer değişkenlerin korelasyonları genellikle daha düşük ve çoğu değişken arasında zayıf ilişkiler bulunuyor.

Korelasyon, iki değişken arasındaki ilişkinin varlığını ve yönünü gösterir ancak nedensellik sağlamaz. Yani, yüksek bir
 korelasyon, bir değişkenin diğerini etkilediği anlamına gelmez"""

# Değişkenler üzerinde herhangi bir aksiyon almadan öne base bir model kurarak inceleyelim

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

Accuracy: 0.77
Recall: 0.706
Precision: 0.59
F1: 0.64
Auc: 0.75

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

plot_importance(rf_model, X)


######################################################################
# Gorev 2 Feature Engineering
######################################################################
# Adım 1:Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumudikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz

def maybe_missing(dataframe, col_name):
    observations = dataframe[dataframe[col_name] == 0].shape[0]
    return observations


for col in num_cols:
    print(col, maybe_missing(df, col))

na_cols = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
na_cols

for col in na_cols:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

"""np.where(): Üç parametre alan bir NumPy fonksiyonudur. İlk parametre bir koşul, ikinci parametre koşul 
doğruysa atanacak değer, üçüncü parametre ise koşul yanlışsa atanacak değerdir."""

#for col in na_cols:
#    df.loc[df[col] == 0, col] = np.NAN

for col in num_cols:
    print(col, maybe_missing(df, col))

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)

# Kayıp değerlerin doldurulması

for col in na_columns:
    df.loc[df[col].isnull(), col] = df[col].median()


df.isnull().sum()

# Aykırı Değer Baskılanması

# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

# Adım 2 :Yeni değişkenler oluşturunuz.

# Yas Kategorisi
df["Age_Cat"] = pd.cut(
    df["Age"], bins=[0, 15, 25, 64, 82],
    labels=["Child", "Young", "Adult", "Senior"])

# BMI Grup
df.loc[(df['BMI'] < 18.5), 'BMI_Group'] = 'Underweight'
df.loc[((df['BMI'] >= 18.5) & (df['BMI'] < 25)), 'BMI_Group'] = 'Normal'
df.loc[((df['BMI'] >= 25) & (df['BMI'] < 30)), 'BMI_Group'] = 'Overweight'
df.loc[(df['BMI'] >= 30), 'BMI_Group'] = 'Obese'
df.head()
# Glikoz Level
def glucose_level(dataframe, col_name="Glucose"):
    if 16 <= dataframe[col_name] <= 140:
        return "Normal"
    else:
        return "Abnormal"
df["Glucose_Level"] = df.apply(glucose_level, axis=1)

# Insulin Level
def insulin_level(dataframe):
    if dataframe["Insulin"] <= 100:
        return "Normal"
    if dataframe["Insulin"] > 100 and dataframe["Insulin"] <= 126:
        return "Prediabetes"
    elif dataframe["Insulin"] > 126:
        return "Diabetes"
df["Insulin_Level"] = df.apply(insulin_level, axis=1)

# Tansiyon Level
def bloodpressure_level(dataframe):
    if dataframe["BloodPressure"] <= 79:
        return "Normal"
    if dataframe["BloodPressure"] > 79 and dataframe["BloodPressure"] <= 89:
        return "Prehypertension"
    elif dataframe["BloodPressure"] > 89:
        return "Hypertension"
df["Bloodpressure_Level"] = df.apply(bloodpressure_level, axis=1)

# Beden kitle endeksine göre şeker seviyesi
df["glucose_per_bmi"] = df["Glucose"] / df["BMI"]

# Yaşa göre insülin seviyesi
df["insulin_per_age"] = df["Insulin"] / df["Age"]

df.head()

# Adım 3: Encoding işlemlerini yapınız

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2 and col not in ["OUTCOME"]]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()
# One-Hot Encoding İşlemi
# cat_cols değişkenlerimiz içerisinde hem glikoz seviyesi hem de hedef değişkenimiz bulunduğu için bunların olmadığı ve ohe yapacağımız değişkenleri seçiyoruz
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
ohe_cols

df.head()

# Adım 4:Numerik değişkenler için standartlaştırma yapınız.

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

# Adım 5:Model oluşturunuz.

y = df["Outcome"]
X = df.drop(["Outcome",'BMI','Insulin','Glucose','BloodPressure','Age'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Ozellik mühendisliği öncesi

# * Accuracy: 0.77
# * Recall: 0.706
# * Precision: 0.59
# * F1: 0.64
# * Auc: 0.75

# Ozellik mühendisliği sonrası

# * Accuracy: 0.77
# * Recall: 0.726
# * Precision: 0.56
# * F1: 0.63
# * Auc: 0.76



def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    #print(feature_imp.sort_values("Value",ascending=False))
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

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(X.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdPu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    plot_roc_curve,
)

confusion_matrix(y_test, y_pred)
"""
Accuracy (Doğruluk): Modelin tüm tahminlerinin ne kadarının doğru olduğunu gösterir. 
Bu durumda, modelin tahminlerinin %77'si doğru.

Recall (Duyarlılık veya Geri Çağırma): Pozitif sınıfı doğru bir şekilde ne kadar iyi 
tespit ettiğini gösterir. Gerçek pozitifleri (hasta olan ve model tarafından hasta olarak 
tahmin edilenler) tüm gerçek pozitiflerin (gerçekten hasta olanların tümü) oranına eşittir. 
Bu örnekte, model gerçek pozitif durumların yaklaşık %70.6'sını doğru tahmin etmiş.

Precision (Kesinlik): Pozitif olarak tahmin edilen durumların ne kadarının gerçekten pozitif 
olduğunu gösterir. Modelin pozitif dediği durumların %59'u gerçekten pozitif.

F1 Score: Precision ve recallun harmonik ortalamasıdır ve her ikisini dengeli bir şekilde göz önünde 
bulundurur. Kesinlik ve duyarlılık arasındaki dengeyi ölçer. F1 skoru, özellikle sınıflar dengesiz olduğunda 
(örneğin bir sınıf diğerlerinden çok daha az örneğe sahipse) önemlidir. Bu model için F1 skoru 0.64tür, yani 
kesinlik ve duyarlılık arasında dengeli bir performans sergiliyor.

AUC (Area Under the ROC Curve): ROC eğrisi (Receiver Operating Characteristic curve) altındaki alanın (AUC) 
değeridir. AUC, modelin rastgele bir pozitif örneği rastgele bir negatif örneğe tercih etme olasılığını ölçer. 
Değerler 0.5 ile 1.0 arasında değişir; 1.0 mükemmel sınıflandırma anlamına gelirken, 0.5 ise modelin rastgele 
tahminler yaptığını gösterir. Bu modelin AUC skoru 0.75('tir, bu da modelin iyi ancak mükemmel olmayan bir ayırt 
edici güce sahip olduğunu gösterir.)

Bu metrikler, modelin performansını değerlendirirken dikkate almanız gereken farklı yönleri temsil eder ve 
genellikle modeli farklı durumlara göre ayarlamak için kullanılır. Örneğin, tıbbi teşhis gibi durumlarda, 
yüksek recall (düşük False Negative oranı) daha önemli olabilir, çünkü gerçek pozitif durumların gözden 
kaçması ciddi sonuçlara yol açabilir."""

