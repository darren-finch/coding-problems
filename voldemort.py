#!/user/bin/env python3
import os

files = []

for file in os.listdir():
	if file == "voldemort.py" or file == "thekey.key":
		continue
	if os.path.isfile(file):
		files.append(file)

print(files)
