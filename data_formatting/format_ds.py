import numpy as np
import pandas as pd
import glob
import re
import multiprocessing as mp
import os
from config import j

ls = glob.glob(j('Dump/*.dump'))

def make_entry_from_line(line):
    _id, _type, x, y, z, vx, vy, vz, count = re.split(r'\s+', line.lstrip().rstrip())
    return {
        'id': int(_id),
        'type': int(_type),
        'x': float(x),
        'y': float(y),
        'z': float(z),
        'vx': float(vx),
        'vy': float(vy),
        'vz': float(vz),
        'count': float(count)
    }

def make_csv_from_dump(filename):
    data = []
    with open(filename, 'r') as f:
        cnt = 0
        for line in f:
            cnt += 1
            if cnt > 9:
                new_entry = make_entry_from_line(line)
                data.append(new_entry)

    df = pd.DataFrame(data).set_index('id')
    output_dir = j('CSV')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, os.path.basename(filename)[:-5] + '.csv')
    df.to_csv(output_filename)
    print(output_filename)

with mp.Pool() as pool:
    pool.map(make_csv_from_dump, ls)