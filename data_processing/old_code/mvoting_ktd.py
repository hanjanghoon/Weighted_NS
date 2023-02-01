# usage: python mvoting_ktd.py result.json test1.json test2.json test3.json ...

import sys
import json

outPath = sys.argv[1]
argInPaths = sys.argv[2:]
inPaths = []
weights = []
w = 1.0
for arg in argInPaths:
    print(arg, w)
    inPaths.append(arg)
    weights.append(w)
    w *= 0.99
    

data = []
for fpath in inPaths:
    with open(fpath) as f:
        _data = json.load(f)
        data.append(_data)


voted_data = []
num_samples = len(_data)
for i in range(num_samples):
    _agg = {True:0, False:0}
    for _data, _w in zip(data, weights):
        _tf = _data[i]["target"]
        _agg[_tf] += _w
    if _agg[True] > _agg[False]:
        voted_data.append({"target": True})
    else:
        voted_data.append({"target": False})

with open(outPath, "w") as f:
    print("majority voting results", outPath)
    json.dump(voted_data, f, indent=2)