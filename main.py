import pandas as pd

columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'snow',
           'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']

df = pd.read_csv('data/12375.csv', header=None, names=columns)
df = df.drop(columns=['tsun', 'coco'])

df = df[(df['date'] >= "2020-05-15") & (df['date'] <= "2025-05-20")]
df['date'] = pd.to_datetime(df['date'])
df['snow'] = df['snow'].fillna(0)
df['prcp'] = df['prcp'].fillna(0)
df['wpgt'] = df['wpgt'].ffill()

print(df.isnull().sum())
# print(df)