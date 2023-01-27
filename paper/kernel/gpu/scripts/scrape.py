# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import glob
import sys

def extract(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x for x in lines if x.strip() != ""]
        try:
            lines[-1] = lines[-1].replace("inf", "-1")
            z = eval(lines[-1])
            if type(z) == dict:
                return z
            return None
        except:
            return None

fs = glob.glob(sys.argv[1]+"/*")
ds = [extract(x) for x in fs]
ds = [x for x in ds if x is not None]

ks = [str(k) for k,v in sorted(ds[0].items(), key=lambda x:x[0])]
print(",".join(ks))
for d in ds:
    if d is None:
        continue
    kvs = []
    for k,v in sorted(d.items(), key=lambda x: x[0]):
        kvs.append(str(v))
    print(",".join(kvs))
