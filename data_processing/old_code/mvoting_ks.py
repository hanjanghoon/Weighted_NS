# usage: python mvoting_ks.py result.json test1.json test2.json test3.json ...

import sys
import json

outPath = sys.argv[1]
argInPaths = sys.argv[2:]
inPaths = []
for arg in argInPaths:
    print(arg)
    inPaths.append(arg)

data = []
for fpath in inPaths:
    with open(fpath) as f:
        _data = json.load(f)
        data.append(_data)


voted_data = []
for i in range(len(_data)):
    item2id = {}
    id2item = {}
    _id = 0
    _agg = {}
    for _w, _data in enumerate(data):
        _tf = _data[i]
        if _tf["target"]:
            for k, item in enumerate(_tf["knowledge"]):
                str_item = str(item)
                if not str_item in item2id:
                    item2id[str_item] = _id
                    id2item[_id] = item
                    _id += 1
                if not item2id[str_item] in _agg:
                    _agg[item2id[str_item]] = 0
                _agg[item2id[str_item]] += (0.99 ** _w) * (0.9 ** k)
        else:
            break
        
    if _agg:
        _agg_know_list = [id2item[_id] for _id, v in sorted(_agg.items(), key=lambda item: item[1])]
        _agg_know_list = _agg_know_list[::-1]
        _res = {"target":True, "knowledge":_agg_know_list[:5]}
    else:
        _res = {"target":False}
    voted_data.append(_res)


with open(outPath, "w") as f:
    print("majority voting results", outPath)
    json.dump(voted_data, f, indent=2)