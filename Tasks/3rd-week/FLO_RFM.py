import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

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

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.

dataframe = pd.read_csv("/Users/mrpurtas/Downloads/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()
           # 2. Veri setinde
                     # a. İlk 10 gözlem,
                     df.head(10)
                     # b. Değişken isimleri,
                     df.columns
                     # c. Betimsel istatistik,
                     df.describe()
                     df.info()
                     # d. Boş değer,
                     df.isnull().sum()
                     # e. Değişken tipleri, incelemesi yapınız.
                     df.dtypes
                     df["master_id"].nunique()
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           df["total_order_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_online"]
           df["total_order_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
           df.dtypes
            date_time = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
            for col in date_time:
               df[col] = pd.to_datetime(df[col])
           # 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
           channel_analy = df.groupby("order_channel").agg({"master_id": "count", "total_order_number": "sum", "total_order_price": "sum"})
           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
           top_10_revenue = df.groupby("master_id").agg({"total_order_price": "sum"}).sort_values(by="total_order_price", ascending=False).head(10)
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
             top_10_order_customer = df.groupby("master_id").agg({"total_order_number": "sum"}).sort_values(by="total_order_number", ascending=False).head(10)
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def create_data_prep(dataframe):
    df = dataframe.copy()
    df["total_order_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_order_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_online"]
    date_time = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for col in date_time:
        df[col] = pd.to_datetime(df[col])
    channel_analy = df.groupby("order_channel").agg({"master_id": "count", "total_order_number": "sum", "total_order_price": "sum"})
    top_10_revenue = df.groupby("master_id").agg({"total_order_price": "sum"}).sort_values(by= "total_order_price", ascending = False).head(10)
    top_10_order_customer = df.groupby("master_id").agg({"total_order_number": "sum"}).sort_values(by= "total_order_number", ascending=False).head(10)
    return df, channel_analy, top_10_order_customer, top_10_revenue

create_data_prep(dataframe)
df.columns
df.head()

# GÖREV 2: RFM Metriklerinin Hesaplanması

df["last_order_date"].max()
today_date = dt.datetime(2021,6, 1)
df.dtypes

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                   "total_order_number": lambda y: y.sum(),
                                   "total_order_price": lambda z: z.sum()})
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
rfm.head

# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

           rfm[["recency", "frequency", "monetary", "segment"]].groupby("segment").agg("mean")
           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ortalama 250 TL üzeri ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

               df1 = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index
               df1_df = pd.DataFrame(df1)

               df.set_index("master_id", inplace=True)

               df2 = df[(df["total_order_number"] > 250) & (df["interested_in_categories_12"].str.contains("KADIN"))]
               df2_df = pd.DataFrame(df2)

result = pd.concat([df1_df, df2_df]).drop_duplicates(keep=False)
result.to_csv("_id.cvs")

                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
                   # olarak kaydediniz.
erkek_cocuk = df[df["interested_in_categories_12"].str.contains("ERKEK | COCUK")].index
hedef_kitle = rfm[rfm["segment"].isin(["cant_loose", "new_customers", "about_to_sleep"])].index
erkek_cocuk_hedef = erkek_cocuk[erkek_cocuk.isin(hedef_kitle)]

result2 = pd.DataFrame(erkek_cocuk_hedef)

result2.to_csv("_ids.csv")

# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def create_data_prep(dataframe):
    df = dataframe.copy()
    df["total_order_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_order_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_online"]
    date_time = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for col in date_time:
        df[col] = pd.to_datetime(df[col])
    channel_analy = df.groupby("order_channel").agg({"master_id": "count", "total_order_number": "sum", "total_order_price": "sum"})
    top_10_revenue = df.groupby("master_id").agg({"total_order_price": "sum"}).sort_values(by= "total_order_price", ascending = False).head(10)
    top_10_order_customer = df.groupby("master_id").agg({"total_order_number": "sum"}).sort_values(by= "total_order_number", ascending=False).head(10)
    return df, channel_analy, top_10_order_customer, top_10_revenue
def rfm_prep(df, csv=False):
        today_date = dt.datetime(2021, 6, 1)
        rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                           "total_order_number": lambda y: y.sum(),
                                           "total_order_price": lambda z: z.sum()})
        rfm.columns = ["recency", "frequency", "monetary"]
        rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
        rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])
        rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
        seg_map = {
            r'[1-2][1-2]': 'hibernating',
            r'[1-2][3-4]': 'at_Risk',
            r'[1-2]5': 'cant_loose',
            r'3[1-2]': 'about_to_sleep',
            r'33': 'need_attention',
            r'[3-4][4-5]': 'loyal_customers',
            r'41': 'promising',
            r'51': 'new_customers',
            r'[4-5][2-3]': 'potential_loyalists',
            r'5[4-5]': 'champions'
        }
        rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
        rfm = rfm[["recency", "frequency", "monetary", "segment"]]
        if csv:
            rfm.to_csv("rfm.csv")
        return rfm

create_data_prep(dataframe)

rfm_prep(df)



###############################################################
# GÖREV 1: Veriyi  Hazırlama ve Anlama (Data Understanding)
###############################################################
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/mrpurtas/Downloads/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()
# 2. Veri setinde
        # a. İlk 10 gözlem,
        # b. Değişken isimleri,
        # c. Boyut,
        # d. Betimsel istatistik,
        # e. Boş değer,
        # f. Değişken tipleri, incelemesi yapınız.



# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["total_order_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df["total_order_number"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]


# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes
date_time = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for col in date_time:
    df[col] = pd.to_datetime(df[col])

# df["last_order_date"] = df["last_order_date"].apply(pd.to_datetime)



# 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız.

df.groupby("order_channel").agg({"master_id": "count", "total_order_price": "sum", 'total_order_number': "sum"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"total_order_price": "sum"}).sort_values(by= 'total_order_price', ascending=False).head(10)


# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"total_order_number": "sum"}).sort_values(by= 'total_order_number', ascending=False).head(10)



# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def create_prep(dataframe):
    df= dataframe.copy()
    df["total_order_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    df["total_order_number"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    date_time = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for col in date_time:
        df[col] = pd.to_datetime(df[col])
    resulttt = df.groupby("order_channel").agg({"master_id": "count", "total_order_price": "sum", 'total_order_number': "sum"})
    top_10_revenue = df.groupby("master_id").agg({"total_order_price": "sum"}).sort_values(by= 'total_order_price', ascending=False).head(10)
    top_10_order_customer = df.groupby("master_id").agg({"total_order_number": "sum"}).sort_values(by= 'total_order_number', ascending=False).head(10)
    return df, resulttt, top_10_revenue, top_10_order_customer

create_prep(dataframe)
###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)


# customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days, "total_order_number": lambda y: y.sum(), "total_order_price": lambda z: z.sum()})
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()
###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################
#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi
rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels= [5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method= "first"), q=5, labels= [1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels= [1, 2, 3, 4, 5])
rfm.head()

# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlama ve  tanımlanan seg_map yardımı ile RF_SCORE'u segmentlere çevirme
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm.head()

rfm = rfm[["recency", "frequency", "monetary", "segment"]]
rfm.head()
###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm.groupby("segment").agg({'recency': "mean", "frequency": "mean", "monetary": "mean"})


# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.

df11 = pd.DataFrame(df1)
df1 = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index
df.head()
df.set_index("master_id", inplace=True)
df.head()
df22 = pd.DataFrame(df2)
df2 = df[df["interested_in_categories_12"].str.contains("KADIN")].index

df1.t

kesisim_df = pd.concat([df11, df22]).drop_duplicates(keep=False)

kesisim_df.to_csv("_idd.csv")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliior. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.

erk_coc = df[df["interested_in_categories_12"].str.contains("ERKEK|COCUK")].index
iyi_mus = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index

sonuc = erk_coc[erk_coc.isin(iyi_mus)]

ids_csv = pd.DataFrame(sonuc)

ids_csv.to_csv("_ids.csv")