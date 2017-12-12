# -*- coding: utf-8 -*-
try:
    import simplejson as json
except ImportError:
    import json

def read_list(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data

def write_list(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)