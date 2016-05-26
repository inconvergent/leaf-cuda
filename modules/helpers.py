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

def link_sort(links):

  curr = links[0,0]
  first = curr
  order = [first]

  while True:

    a = links[curr,0]
    b = links[curr,1]

    if a != curr:
      curr = a
    else:
      curr = b

    order.append(curr)

    if curr == first:
      order.append(a)
      break

  return order
