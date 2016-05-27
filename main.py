#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import division


def get_wrap(dl, colors, render_steps=10, export_steps=10):

  from time import time
  from numpy.random import random

  t0 = time()

  from fn import Fn
  fn = Fn(prefix='./res/')

  rndcolors = random((dl.nz2,3))
  rndcolors[:,0] *= 0.1

  def wrap(render):

    dl.step()

    if dl.itt % render_steps == 0:

      snum = dl.snum
      vnum = dl.vnum
      zone = dl.zone[:snum]

      sxy = dl.sxy[:snum,:]
      vxy = dl.vxy[:vnum,:]
      vs = dl.vs[:vnum]
      tmp = dl.tmp[:vnum]
      print(dl.tmp)

      print('itt', dl.itt, 'snum', snum, 'vnum', vnum, 'time', time()-t0)

      render.clear_canvas()
      render.set_line_width(dl.one)

      for j, (x,y) in enumerate(sxy):
        rgba = list(rndcolors[zone[j]%len(rndcolors),:]) + [0.3]
        render.set_front(rgba)
        render.circle(x, y, 2*dl.one, fill=True)

      render.set_front(colors['red'])
      for i, (x,y) in enumerate(vxy):
        render.circle(x, y, 2*dl.one, fill=True)

      for i in xrange(vnum):
        j = vs[i]
        t = tmp[i]
        if j < 0:
          print(i,j,t,'WARNING: no nearby source found')
          continue
        else:
          print(i,j,t,'OK')
        render.line(sxy[j,0], sxy[j,1], vxy[i,0], vxy[i,1])

    # if dl.itt % export_steps == 0:

      name = fn.name()
      render.write_to_png(name+'.png')
      # # export('lattice', name+'.2obj', vertices, edges=edges)

    return False

  return wrap



def main():

  from modules.leaf import Leaf
  from render.render import Animate
  # from numpy.random import random
  from numpy import array

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.6],
    'cyan': [0,0.6,0.6,0.6],
    'red': [0.7,0.0,0.0,0.7],
    'light': [0,0,0,0.2],
  }

  threads = 512

  render_steps = 1
  export_steps = 100000

  size = 512*2
  one = 1.0/size

  init_sources = 20000
  init_veins = array([[0.5,0.5]])

  stp = one*0.4

  DL = Leaf(
    size,
    stp,
    init_sources,
    init_veins,
    threads = threads
  )

  wrap = get_wrap(
    DL,
    colors,
    export_steps=export_steps,
    render_steps=render_steps
  )

  render = Animate(size, colors['back'], colors['front'], wrap)
  render.start()


if __name__ == '__main__':

  main()

