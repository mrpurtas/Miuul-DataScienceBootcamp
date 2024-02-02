import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import numpy as np
from scipy import stats

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#pip install yellowbrick
import pandas as pd
import yellowbrick
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings

df = pd.read_csv('datasets/flo_data_20k.csv')

df.info()
df.columns
df.head()

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[ns]').dt.days # en son kaç gün önce alışveriş yaptı
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[ns]').dt.days

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()


#Bu fonksiyon, verilen bir DataFrame (df_skew) ve sütun adı (column) için çarpıklık değerini hesaplıyor
# ve sütunun dağılımını görselleştiriyor.
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()


model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()


sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()


k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments=k_means.labels_+1
segments

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()


final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max","std"],
                                  "order_num_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_online":["mean","min","max","std"],
                                  "recency":["mean","min","max","std"],
                                  "tenure":["mean","min","max","count","std"]})



hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()

hc_average = linkage(model_df, 'average')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.7, color='r', linestyle='-.')
plt.show()


hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)


final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df.loc[:, "segment"] = segments + 1
final_df.head()

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max","std"],
                                  "order_num_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_offline":["mean","min","max","std"],
                                  "customer_value_total_ever_online":["mean","min","max","std"],
                                  "recency":["mean","min","max","std"],
                                  "tenure":["mean","min","max","count","std"]})