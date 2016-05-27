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

      sxy = dl.sxy[:snum,:]
      vxy = dl.vxy[:vnum,:]
      sv = dl.sv[:snum]
      tmp = dl.tmp[:snum]

      print('itt', dl.itt, 'snum', snum, 'vnum', vnum, 'time', time()-t0)

      render.clear_canvas()
      render.set_line_width(dl.one)

      # sources
      # render.set_front(colors['red'])
      # for x,y in sxy:
        # render.circle(x, y, 2*dl.one, fill=True)

      # veins
      render.set_front(colors['front'])
      for x,y in vxy:
        render.circle(x, y, 3*dl.one, fill=True)

      # nearby
      render.set_front(colors['red'])
      for s in xrange(snum):
        v = sv[s]
        if v<0 or s<0:
          print('WARNING: v', v, 's', s)
          continue
        render.line(sxy[s,0], sxy[s,1], vxy[v,0], vxy[v,1])
        print('OK: v', v, 's', s)

      print(tmp)
      print(sv, (sv>-1).sum(), len(sv))
      print(dl.nz, dl.rad)

    # if dl.itt % export_steps == 0:
      name = fn.name()
      render.write_to_png(name+'.png')
      # # export('lattice', name+'.2obj', vertices, edges=edges)


    return False

  return wrap



def main():

  from modules.leaf import Leaf
  from render.render import Animate
  from numpy.random import random
  from numpy import array

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.6],
    'cyan': [0,0.6,0.6,0.6],
    'red': [0.7,0.0,0.0,0.3],
    'light': [0,0,0,0.2],
  }

  threads = 512

  render_steps = 1
  export_steps = 100000

  size = 512*2
  one = 1.0/size

  rad = 0.1
  sources_dst = rad*0.01

  init_sources = 20000
  # init_veins = array([[0.5,0.5], [0.1,0.1]])
  init_veins = random((1000,2))

  stp = one*0.4

  DL = Leaf(
    size,
    stp,
    init_sources,
    init_veins,
    rad,
    sources_dst,
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

