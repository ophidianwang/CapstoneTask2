#!/usr/bin/python

import glob
import shutil
import random

cat_list = glob.glob("source/*")
cat_size = len(cat_list)

print("source contains " + str(cat_size) + " categories")

if cat_size < 1:
    print "you need to generate the cuisines files 'categories' folder first"

sample_size = min(30, cat_size)
cat_sample = sorted(random.sample(range(cat_size), sample_size))

print("sample on " + str(len(cat_sample)) + " categories")

for i, item in enumerate(cat_list):
    if i >= len(cat_sample):
        break
    li = item.split('/')
    print("#" + str(i) + " copy " + str(item) + " to " + "categories/" + str(li[-1]))
    shutil.copyfile(str(item), "categories/" + str(li[-1]))

print("done")
