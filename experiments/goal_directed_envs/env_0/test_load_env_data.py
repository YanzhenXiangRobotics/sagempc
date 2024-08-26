import pickle
import os
with open(os.path.join(os.path.dirname(__file__), "env_data.pkl")) as f:
    data = pickle.load(f)
print(data)