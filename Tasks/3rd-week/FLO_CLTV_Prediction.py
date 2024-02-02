##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama

# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.
# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df_ = pd.read_csv("/Users/mrpurtas/Desktop/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.describe([0.01,0.25,0.5,0.75,0.99]).T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

    threshold_list = ["order_num_total_ever_online", "order_num_total_ever_offline",
                    "customer_value_total_ever_offline","customer_value_total_ever_online"]

    for col in threshold_list:
        replace_with_thresholds(df, col)

    df["total_customer_value"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_customer_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    date_list = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for col in date_list:
        df[col] = pd.to_datetime(df[col])

        df.dtypes

# GÖREV 2: CLTV Veri Yapısının Oluşturulması

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
df.dtypes
df.head()

df.groupby("master_id").agg({})
cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["frequency"] = df["total_customer_value"]
cltv_df["monetary_cltv_avg"] = df["total_customer_price"] / df["total_customer_value"]



# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.
    # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
    # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"],
                                cltv_df["monetary_cltv_avg"], time=6, freq="W", discount_rate=0.01)
    # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values(by="cltv", ascending=False).head()

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], q=4, labels= ["D", "C", "B", "A"])

# BONUS: Tüm süreci fonksiyonlaştırınız.


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################


# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.


# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.



###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.


# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.


# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.



# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.


# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.


# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.




###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.


# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.








