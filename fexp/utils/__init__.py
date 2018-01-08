# -*- coding: utf-8 -*-
try:
    import simplejson as json
except ImportError:
    import json


def read_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)


def read_list(filename):
    """Reads file with caseids, separated by line.
    """
    f = open(filename, 'r')
    ids = []
    for line in f:
        ids.append(line.strip())
    f.close()

    return ids


def write_list(input_list, filename, append=False):
    """Reads a list of strings and writes the list line by line to a text file."""
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        for line in input_list:
            f.write(line.strip() + '\n')
