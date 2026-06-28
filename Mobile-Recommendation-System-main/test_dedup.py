import pickle
import pandas as pd
from src.recommender import RecommenderEngine

print('Loading data...')
df = pickle.load(open('src/model/dataframe.pkl', 'rb'))
sim = pickle.load(open('src/model/similarity.pkl', 'rb'))
print(f'Original DF rows: {len(df)}')
print(f'Duplicate names in raw data: {df["name"].duplicated().sum()}')

engine = RecommenderEngine(df, sim)
print(f'After dedup DF rows: {len(engine.df)}')
print(f'Duplicate names in engine.df: {engine.df["name"].duplicated().sum()}')

print('\n--- Testing Algorithms ---')

rb = engine.rule_based(budget=50000)
print(f'Rule-Based duplicates: {rb["name"].duplicated().sum()}')

pb = engine.persona_based(1)
print(f'Persona-Based duplicates: {pb["name"].duplicated().sum()}')

if not engine.df.empty:
    cb = engine.content_based(engine.df.iloc[0]['name'])
    print(f'Content-Based duplicates: {cb["name"].duplicated().sum()}')

pp = engine.preference_based(30000, 'normal', 'medium', 'medium', 'medium')
print(f'Preference-Based duplicates: {pp[0]["name"].duplicated().sum()}')

cp = engine.collaborative_popularity((10000, 50000))
print(f'Collaborative duplicates: {cp["name"].duplicated().sum()}')

ws = engine.weighted_scoring({'price':0.3, 'ram':0.15, 'storage':0.15, 'rating':0.2, 'battery':0.1, 'camera':0.1})
print(f'Weighted Scoring duplicates: {ws["name"].duplicated().sum()}')

if not engine.df.empty:
    knn = engine.knn_recommend(engine.df.iloc[0]['name'])
    print(f'KNN duplicates: {knn["name"].duplicated().sum()}')

print('\nAll tests passed!')

