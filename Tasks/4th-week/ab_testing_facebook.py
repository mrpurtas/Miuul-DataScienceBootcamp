#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

control_group = pd.read_excel("/Users/mrpurtas/PycharmProjects/pythonProject/datasets/ab_testing.xlsx", sheet_name="Control Group")
test_group = pd.read_excel("/Users/mrpurtas/PycharmProjects/pythonProject/datasets/ab_testing.xlsx", sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
control_group.describe().T
control_group.dropna(inplace=True)
test_group.describe().T

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df = pd.concat([control_group, test_group], axis=0, ignore_index=True)
control_group["group"] = "control"
test_group["group"] = "test"

df["don_oranı"] = df["Purchase"] / df["Impression"]
df.head()


#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

H0 : M1 = M2
H1 : M1!= M2


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
df[df["group"] == "control"].mean() #control grubunda daha az reklam goruntulenmesıne ragmen daha cok tıklama almıştır
                                    #bu durum kontrol grubundaki musterılere daha dogru reklam cıkartıldıgının gostergesi olabilir.
df[df["group"] == "test"].mean()

df.loc[df["group"] == "control", "Purchase"].mean()
df.loc[df["group"] == "test", "Purchase"].mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################
# H0: Ask as purchase between Maxbidding (A, control) and Avgbidding (B, test) there is no
# statistically significant difference.
# H1: ... there is a difference.

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.5891 dolayısıyla normal dağılır
test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.1541 so normal dağılır
test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "don_oranı"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.8844
test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "don_oranı"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.3980


test_stat, pvalue = levene(df.loc[df["group"] == "test", "Purchase"],
                           df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#0.1083 > 0.005 so h0 reddedilemez yani varyans homojen dagılmıstır

test_stat, pvalue = levene(df.loc[df["group"] == "test", "don_oranı"],
                           df.loc[df["group"] == "control", "don_oranı"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# 0.7868 so h0 reddedilemez yani varyans homojen dagılmıstır

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

#varsayım kontrollerı sonucu iki kosulda sağlandıgı için bagımsız iki orneklem testi (parametrik test) uygulanır
# H0: İki grup arasında istatistiksel olarak anlamlı bir fark yoktur.
# H1: Vardır.
# Two-Sample T-Test, which is a parametric test, is required.



# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#p-value = 0.3493 so h0 reddedilemez
# there is no statistically significant difference between the two methods.

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "don_oranı"],
                              df.loc[df["group"] == "test", "don_oranı"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.0557 so there is no significant difference between the two methods yani reklamların çalışırlığıyla alakalı>
#bir değişim olmamıştır

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# We used independent t-test because we want to determine if there is a
# significant difference between the means of two indepented groups, which may be related in certain features.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
# There is no statistically significant difference between the Control group that was served “maximum bidding” campaign
# and Test group that was served “average bidding” campaign.

#toplam reklam sayısı / satın alma yanı donusum oranı kıyası yapıldıgında da ciddi bi farklılık gozlenmemiştir


####################################
# 2. AB TEST:
####################################
# H0: # H0: ist in terms of gain between Maxbidding (A, control) and Avgbidding (B, test). die.
    # There is no significant difference.
# H0: # H1: ... there is no difference.

# Assumption of Normality: provided.
#################

# H0: Assumption of normal distribution is provided.
# H1:..not provided.


test_stat, pvalue = shapiro(df.loc[df["group"] == "control"]["Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# TTest Stat = 0.9756, p-value = 0.5306

test_stat, pvalue = shapiro(df.loc[df["group"] == "test"]["Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9780, p-value = 0.6163
# Assumption of Variance Homogeneity: provided.


#################

# H0: Variances are homogeneous.
# H1: ... is not.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Earning"],
                           df.loc[df["group"] == "test", "Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.3532, p-value = 0.5540  H0: NOT REJECTED

# AB TEST:

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Earning"],
                              df.loc[df["group"] == "test", "Earning"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = -9.2545, p-value = 0.0000

## p value < 0.05, H0: REJECTED

sns.lineplot(x = "don_oranı", y = "Earning", hue = "group", data = df);
plt.show()


df.loc[df["group"] == "control", "Earning"].mean()
df.loc[df["group"] == "test", "Earning"].mean()

# CONCLUSION:

# Max bidding method targets customers who can be found with the determined max price.
# In this case, users above the specified fee cannot be accessed, but those who fall far below can also be accessed.
# Avg. With bidding, it reaches users from both segments by balancing the above and below the determined price.
# a more strategic advertising fee will be made.
# Therefore, it is also important that the resulting purchase is the result of how many people see the advertisement.
# Although the purchase did not show a statistically significant increase with the new system,
# It is thought that there may be an increase in earnings due to the possible increase in the conversion rate (purchase per click).
# For this reason, which will bring more profit than the result obtained after the 2nd EU test.
# It is recommended to use the AVERAGE BIDDING method.

"""Sonuç:
1. Max Bidding Yöntemi:

Bu yöntem, belirlenen maksimum fiyata kadar olan müşterilere odaklanır.
Belirlenen ücretin üzerindeki kullanıcılara ulaşılamaz ancak çok daha düşük olanlara ulaşmak da mümkündür.

2. Ortalama (Avg) Bidding Yöntemi:

Bu yöntem, belirlenen fiyatın hem üzerindeki hem de altındaki kullanıcılara ulaşarak her iki segmentten de kullanıcılara ulaşmayı amaçlar.
Böylece, daha stratejik bir reklam ücreti belirlenir.
3. Reklamın Etkinliği:

Reklamı gören kaç kişinin sonucunda satın alma yaptığı da önemlidir. Yani, reklamın etkinliğini ölçen bir metrik olan dönüşüm oranı (satın alma başına tıklama oranı) üzerinde bir artış olabilir.
4. İstatistiksel Sonuç:

Yeni sistemle birlikte satın almada istatistiksel olarak anlamlı bir artış gözlenmemiş olsa da, dönüşüm oranında olası bir artış nedeniyle gelirde bir artış olabileceği düşünülmektedir.
5. Öneri:

İkinci AB testinden alınan sonuçtan daha fazla kâr getireceği düşünülen yöntem olan Ortalama Bidding yönteminin kullanılması önerilmektedir.
Özetle, iki farklı reklam teklif yöntemi olan "Max Bidding" ve "Ortalama Bidding" arasındaki farklar ve etkileri değerlendirilmiştir. İstatistiksel sonuçlar, Ortalama Bidding yönteminin daha kârlı olabileceğini gösteriyor ve bu nedenle bu yöntemin kullanılması öneriliyor.

"""
