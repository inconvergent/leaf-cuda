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
  step = dl.step()

  def wrap(render):

    sv = step.next()

    if dl.itt % render_steps == 0:

      snum = dl.snum
      vnum = dl.vnum

      sxy = dl.sxy[:snum,:]
      vxy = dl.vxy[:vnum,:]

      print('itt', dl.itt, 'snum', snum, 'vnum', vnum, 'time', time()-t0)

      render.clear_canvas()
      render.set_line_width(dl.one)

      # sources
      render.set_front(colors['cyan'])
      for x,y in sxy:
        render.circle(x, y, 1.5*dl.one, fill=True)

      # veins
      render.set_front(colors['front'])
      for x,y in vxy:
        render.circle(x, y, 1.5*dl.one, fill=True)

      # nearby
      warnings = 0
      oks = 0
      render.set_front(colors['cyan'])
      for s in xrange(snum):
        v = sv[s]
        if v<0 or s<0:
          # print('WARNING: v', v, 's', s)
          warnings += 1
          continue
        render.line(sxy[s,0], sxy[s,1], vxy[v,0], vxy[v,1])
        oks += 1
      print('snum: ', snum)
      print('vnum: ', vnum)
      print('WARNING: ', warnings)
      print('OK: ', oks)

    if dl.itt % export_steps == 0:
      name = fn.name()
      render.write_to_png(name+'.png')
      # # export('lattice', name+'.2obj', vertices, edges=edges)

    return True

  return wrap



def main():

  from modules.leaf import Leaf
  from render.render import Animate
  from numpy.random import random

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.1],
    'cyan': [0,0.6,0.6,0.3],
    'red': [0.7,0.0,0.0,0.3],
    'light': [0,0,0,0.2],
  }

  threads = 512

  render_steps = 10
  export_steps = 10

  size = 512*2
  one = 1.0/size

  node_rad = 20*one

  area_rad = 10*node_rad
  sources_rad = node_rad
  stp = node_rad*0.1
  kill_rad = node_rad

  init_num_sources = 5000
  init_veins = random((1,2))

  from dddUtils.random import darts_rect
  init_sources = darts_rect(
    init_num_sources,
    0.5,
    0.5,
    0.9,
    0.9,
    sources_rad
  )


  DL = Leaf(
    size,
    stp,
    init_sources,
    init_veins,
    area_rad,
    kill_rad,
    sources_rad,
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

