import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# gets title from db using index
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

# gets index from db using title
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

# reading from csv file
df = pd.read_csv("movie_dataset.csv")
# key features
features = ['keywords','cast','genres','director']

# combine all features
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:", row)

df["combined_features"] = df.apply(combine_features,axis=1)

# new count matrix
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

# computing cos similarity 
cosine_sim = cosine_similarity(count_matrix)

# seeing if user would like to continue
toContinue = True

while toContinue:
    movie_user_likes = input('Which movie do you like: ')
    while True:
        if movie_user_likes == '':
            print("quitting ...")
            exit(1)
        movie_user_likes = movie_user_likes.strip(' ') # remove any extra spaces
        if movie_user_likes in df.original_title.to_numpy():
            # movie valid, move on
            break
        movie_user_likes = input('Movie unknown, please entre a new movie: ')

    # getting index of movie
    movie_index = get_index_from_title(movie_user_likes)
    # list of similar movies
    similar_movies =  list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

    # printing top 50 similar movies
    i=0
    print('\n - Your Top 10 Recommended -\n')
    for element in sorted_similar_movies:
        # skipping first recommendation (same movie as uer input)
        if(i==0):
            i += 1
            continue
        # printing title and number
        print(f'{i}: {get_title_from_index(element[0])}')
        # increment
        i += 1
        # top 10
        if i>10:
            break

    ask = input('\nPress Entre to get more recommendations!')
    if ask == '':
        toContinue = False
# user requested quit application
print('\nquitting ...')