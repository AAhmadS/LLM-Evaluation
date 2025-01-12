import pickle

s = "Overall"
a = "context"
with open('overall_score_column_name.pkl', 'wb') as f:
    pickle.dump(s, f)

with open('content_column_name.pkl', 'wb') as f:
    pickle.dump(a, f)
