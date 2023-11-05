import pandas as pd


df = pd.read_csv('./TextData/dataset_with_larger_description.csv')

genre_dic = {}
label_columns = df['genre'].unique().tolist()
for idx, item in enumerate(df['genre'].unique().tolist()):
  genre_dic[item] = idx

df['label'] = df['genre'].apply(lambda x: genre_dic[x])

df.drop(['movie_id', 'genre'], axis=1, inplace=True)

part_50 = df.sample(frac = 0.5)
rest_part_50 = df.drop(part_50.index)

part_50.to_csv("./TextData/text_first_client.csv", index=False)
rest_part_50.to_csv("./TextData/text_second_client.csv", index=False)

print(part_50.head())
print(rest_part_50.head())