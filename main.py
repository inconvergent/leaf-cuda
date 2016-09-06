#!/usr/bin/python3
#-*- coding: utf-8 -*-


def get_wrap(l, colors, node_rad, render_steps=10, export_steps=10):

  from time import time
  from time import strftime
  from numpy.linalg import norm
  from numpy import sqrt
  from iutils.ioOBJ import export_2d as export


  t0 = time()

  from fn import Fn
  fn = Fn(prefix='./res/')

  step = l.step()
  kill_rad = l.kill_rad
  one = l.one

  def wrap(render):

    final = False
    vs_xy = []

    try:
      vs_xy = next(step)
    except StopIteration:
      final = True

    if (l.itt % render_steps == 0) or final:

      vnum = l.vnum
      edges = l.edges[:l.enum,:]

      width = l.width_calc(scale=6.5*node_rad, min_width=node_rad, po=0.2)

      vxy = l.vxy[:vnum,:]

      zsize = len(l.zone_node)
      print(strftime("%Y-%m-%d %H:%M:%S"), 'itt', l.itt,
          'snum', l.snum, 'vnum', vnum, 'zone', zsize, 'time', time()-t0)

      render.clear_canvas()

      # # nearby
      # render.set_front(colors['cyan'])
      # for v,s in vs_xy:
        # render.line(v[0], v[1], s[0], s[1])

      # veins
      # render.set_front(colors['vein'])
      # for i,(x,y) in enumerate(vxy):
        # r = node_rad
        # # r = (((max_gen-gen[i])/max_gen)**1.1)*node_rad
        # render.circle(x, y, r, fill=True)

      # edges
      render.set_front(colors['vein'])
      for ee in edges:
        xy = vxy[ee, :]
        # r = node_rad
        r = width[ee[1]]
        try:
          render.circles(xy[0,0], xy[0,1], xy[1,0], xy[1,1], r, nmin=3)
        except Exception as e:
          print('WARNING', str(e))

      ## sources
      # render.set_front(colors['red'])
      # for x,y in l.sxy:
        # render.circle(x, y, one, fill=True)

      # for x,y in l.sxy[l.smask]:
        # render.circle(x, y, kill_rad, fill=False)

    if (l.itt % export_steps == 0) or final:
      name = fn.name()
      render.write_to_png(name+'.png')
      # export('leaf', name+'.2obj', vxy)

    if final:
      return False

    # raw_input('')
    return True

  return wrap



def main():

  from modules.leaf import LeafClosed as Leaf
  from iutils.render import Animate
  from numpy.random import random
  from numpy import array

  colors = {
    'back': [1,1,1,1],
    'front': [0,0,0,0.3],
    'vein': [0,0,0,0.9],
    'edge': [0,0,0,0.6],
    'cyan': [0,0.6,0.6,0.3],
    'red': [0.7,0.0,0.0,0.8],
    'blue': [0.0,0.0,0.7,0.8],
    'light': [0,0,0,0.2],
  }

  threads = 512

  render_steps = 3
  export_steps = 3

  size = 512
  one = 1.0/size

  node_rad = 0.5*one

  area_rad = 20*node_rad
  stp = node_rad*2
  kill_rad = 2*stp
  sources_dst = 2*kill_rad

  # init_num_sources = 4
  # init_veins = 0.2+0.6*random((init_num_sources,2))
  init_veins = array([[0.5, 0.5]])

  init_num_sources = 10000

  from iutils.random import darts
  init_sources = darts(init_num_sources, 0.5, 0.5, 0.45, sources_dst)
  # from iutils.random import darts_rect
  # init_sources = darts_rect(init_num_sources, 0.5, 0.5, 0.95, 0.95, sources_dst)

  L = Leaf(
    size,
    stp,
    init_sources,
    init_veins,
    area_rad,
    kill_rad,
    threads = threads
  )
  print('nz', L.nz)
  print('dens', L.sv_leap)

  wrap = get_wrap(
    L,
    colors,
    node_rad=node_rad,
    export_steps=export_steps,
    render_steps=render_steps
  )

  render = Animate(size, colors['back'], colors['front'], wrap)
  render.set_line_width(L.one*2)
  render.start()


if __name__ == '__main__':

  main()

