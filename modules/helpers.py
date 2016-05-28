#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function

def load_kernel(fn, name, subs={}):

  from pycuda.compiler import SourceModule

  with open(fn, 'r') as f:
    kernel = f.read()

  for k,v in subs.iteritems():
    kernel = kernel.replace(k, str(v))

  mod = SourceModule(kernel)
  return mod.get_function(name)

def show_open(render, snum, sxy, vxy, sv):

  for s in xrange(snum):
    v = sv[s]
    if v<0 or s<0:
      continue
    render.line(sxy[s,0], sxy[s,1], vxy[v,0], vxy[v,1])

