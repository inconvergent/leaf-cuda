#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import division


def get_wrap(l, colors, render_steps=10, export_steps=10):

  from time import time
  from time import strftime
  from numpy import pi
  from modules.helpers import show_closed
  from numpy.random import random

  t0 = time()

  from fn import Fn
  fn = Fn(prefix='./res/')

  rndcolors = random((l.nz2,3))
  rndcolors[:,0] *= 0.1
  step = l.step()
  one = l.one

  def wrap(render):

    final = False

    try:
      vs_dict = step.next()
    except StopIteration:
      final = True

    if (l.itt % render_steps == 0) or final:

      snum = l.snum
      vnum = l.vnum

      sxy = l.sxy[:snum,:]
      vxy = l.vxy[:vnum,:]

      zsize = len(l.zone_node)
      print(strftime("%Y-%m-%d %H:%M:%S"), 'itt', l.itt,
          'snum', snum, 'vnum', vnum, 'zone', zsize, 'time', time()-t0)

      render.clear_canvas()
      r = one*10

      # veins
      render.set_front(colors['front'])
      for i,(x,y) in enumerate(vxy):
        render.circle(x, y, r, fill=True)
        # render.ctx.arc(x,y,r,0,pi*(1.+random()))
        # render.ctx.fill()

      render.set_front(colors['red'])
      for i in l.parents:
        x,y = vxy[i,:]
        render.circle(x, y, 0.5*r, fill=True)

      render.set_front(colors['blue'])
      for i in l.children:
        x,y = vxy[i,:]
        render.circle(x, y, 0.5*r, fill=True)

      # # sources
      render.set_front(colors['cyan'])
      for x,y in sxy:
        render.circle(x, y, 0.5*r, fill=True)

      # # nearby
      render.set_front(colors['front'])
      show_closed(render, sxy, vxy, vs_dict)

    if (l.itt % export_steps == 0) or final:
      name = fn.name()
      render.write_to_png(name+'.png')

    if final:
      return False

    raw_input()

    return True

  return wrap



def main():

  from modules.leaf import LeafClosed as Leaf
  from render.render import Animate
  from numpy.random import random
  from numpy import array

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.1],
    'cyan': [0,0.6,0.6,0.3],
    'red': [0.7,0.0,0.0,0.3],
    'blue': [0.0,0.0,0.7,0.3],
    'light': [0,0,0,0.2],
  }

  threads = 512

  render_steps = 1
  export_steps = 2000

  size = 1024
  one = 1.0/size

  node_rad = 20*one

  area_rad = 5*node_rad
  sources_rad = 2*node_rad
  stp = node_rad
  kill_rad = node_rad

  # init_num_sources = 4
  # init_veins = 0.2+0.6*random((init_num_sources,2))
  init_veins = array([[0.5]*2])

  init_num_sources = 50000

  from dddUtils.random import darts
  init_sources = darts(init_num_sources, 0.5, 0.5, 0.45, sources_rad)
  # from dddUtils.random import darts_rect
  # init_sources = darts_rect(init_num_sources, 0.5, 0.5, 0.95, 0.95, sources_rad)

  L = Leaf(
    size,
    stp,
    init_sources,
    init_veins,
    area_rad,
    kill_rad,
    sources_rad,
    threads = threads
  )
  print('nz', L.nz)
  print('dens', L.sv_leap)

  wrap = get_wrap(
    L,
    colors,
    export_steps=export_steps,
    render_steps=render_steps
  )

  render = Animate(size, colors['back'], colors['front'], wrap)
  render.set_line_width(L.one)
  render.start()


if __name__ == '__main__':

  main()

