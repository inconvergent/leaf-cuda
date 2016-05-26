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

  colors = random((10,3))

  def wrap(render):

    dl.step()

    if dl.itt % render_steps == 0:

      snum = dl.snum
      vnum = dl.vnum
      zone = dl.zone[:snum]
      nz2 = dl.nz2

      sxy = dl.sxy[:snum,:]

      print('itt', dl.itt, 'snum', snum, 'vnum', vnum, 'time', time()-t0)

      render.clear_canvas()
      render.set_line_width(dl.one)

      ## dots
      for i, (x,y) in enumerate(sxy):
        # c = zone[i]/nz2
        # print(c)
        # rgba = 4*[c]
        # rgba[3] = 1

        rgba = list(colors[zone[i]%len(colors),:]) + [1]
        render.set_front(rgba)
        render.circle(x, y, 2*dl.one, fill=True)

    # if dl.itt % export_steps == 0:

      name = fn.name()
      render.write_to_png(name+'.png')
      # # export('lattice', name+'.2obj', vertices, edges=edges)

    return False

  return wrap



def main():

  from modules.leaf import Leaf
  from render.render import Animate

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.6],
    'cyan': [0,0.6,0.6,0.6],
    'light': [0,0,0,0.2],
  }

  threads = 512

  render_steps = 1
  export_steps = 100000

  size = 512*2
  one = 1.0/size

  init_sources = 200000

  stp = one*0.4

  DL = Leaf(
    size,
    stp,
    init_sources,
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

