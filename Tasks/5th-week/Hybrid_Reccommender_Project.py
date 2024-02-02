import pandas as pd
pd.set_option('display.max_columns', 500)

pd.set_option('display.max_columns', 5)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################


# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv('datasets/hybrid_recommender/movie.csv')
rating = pd.read_csv('datasets/hybrid_recommender/rating.csv')


# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti



# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df = rating.merge(movie[["movieId", "genres"]], how='left')

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

rating_counts = pd.DataFrame(df["movieId"].value_counts())
# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz

rare_movies = rating_counts[rating_counts["movieId"] < 5000].index.tolist()

common_movies = df[~df["movieId"].isin(rare_movies)]


# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index='userId', columns='movieId', values='rating')
# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım



#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=15).values)
6835
# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.columns
# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched = random_user_df.columns[~random_user_df.isna().all()].tolist()

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched]
len(movies_watched_df)
# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.sort_values(by="movie_count", ascending=False)
# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]


#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
df.head()
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])
# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()
corr_df.sort_values(by="corr", ascending=False)
#corr_df[corr_df["user_id_1"] == random_user]



# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["corr"] > 0.65) & (corr_df["user_id_1"] == random_user)][["user_id_2", "corr"]].reset_index(drop=True)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]
top_users_ratings.sort_values(by="weighted_rating", ascending=False)
# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df.reset_index(inplace=True)
# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

movies_to_be_recommend = recommendation_df.sort_values(by="weighted_rating", ascending=False).head(5)


# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')

movies_to_be_recommend.merge(movie[["movieId", "title"]])


#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/hybrid_recommender/rating.csv')

user_movie_df.head()
# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"].iloc[0]

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
selected_movie_rating = user_movie_df[1]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

correlations = user_movie_df.corrwith(selected_movie_rating).dropna().sort_values(ascending=False)
corr_df = pd.DataFrame(correlations)
corr_df.index.names = ["movieId"]
corr_df.reset_index(inplace=True)
corr_df.columns = ["movieId", "corr"]


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.


corr_df = corr_df.sort_values(by= "corr", ascending=False).iloc[1:6, :]
movie.head()

corr_df.merge(movie, on="movieId", how="inner")

  movieId      corr                  title                                            genres
0     3114  0.739854     Toy Story 2 (1999)       Adventure|Animation|Children|Comedy|Fantasy
1    78499  0.588510     Toy Story 3 (2010)  Adventure|Animation|Children|Comedy|Fantasy|IMAX
2     2355  0.528696   Bug's Life, A (1998)               Adventure|Animation|Children|Comedy
3     4886  0.520558  Monsters, Inc. (2001)       Adventure|Animation|Children|Comedy|Fantasy
4     6377  0.504607    Finding Nemo (2003)               Adventure|Animation|Children|Comedy
