#!/usr/bin/env python

import time

t0 = time.clock()

for i in range(10000):
	print "hi"

print time.clock() - t0
