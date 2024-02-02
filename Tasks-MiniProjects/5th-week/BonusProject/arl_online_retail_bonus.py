!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

#Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)

df = df[~df["StockCode"].astype(str).isin(["POST"])]
df.shape
df.isnull().sum()

#Adım 3: Boş değer içeren gözlem birimlerini drop ediniz
df.dropna(inplace=True)

#Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)

df = df[~df["Invoice"].str.contains("C", na=False)]

#Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.

df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]

#Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

#Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.
df_ger = df[df["Country"] == "Germany"]
df_ger = df_ger.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(lambda x:1 if x > 0 else 0)

#Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.

frequent_itemsets = apriori(df_ger,
                            min_support=0.009,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.009)

rules.sort_values("support", ascending=False)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                consequent = list(sorted_rules.iloc[i]["consequents"])[0]
                if consequent not in recommendation_list:
                    recommendation_list.append(consequent)
    return recommendation_list[0:rec_count]


arl_recommender(rules, 23235, 5)
arl_recommender(rules,  21987, 5)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)