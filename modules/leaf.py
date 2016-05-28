# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division


from numpy import pi
from numpy import zeros
from numpy import sin
from numpy import cos
from numpy import sqrt
from numpy.random import random

from numpy import float32 as npfloat
from numpy import int32 as npint


TWOPI = pi*2
PI = pi



class Leaf(object):

  def __init__(
      self,
      size,
      stp,
      num_sources,
      veins,
      area_rad,
      kill_rad,
      sources_rad,
      threads = 256,
      nmax = 1000000
    ):

    self.itt = 0

    self.area_rad = area_rad
    self.kill_rad = kill_rad
    self.threads = threads
    self.nmax = nmax
    self.size = size

    self.one = 1.0/size
    self.stp = stp
    self.init_sources = num_sources

    self.sources_rad = sources_rad

    self.__init()
    self.__init_sources(self.init_sources)
    self.__init_veins(veins)
    self.__cuda_init()

  def __init(self):

    self.vnum = 0
    self.snum = 0

    nz = int(1.0/(2*self.area_rad))
    self.nz = nz

    self.nz2 = self.nz**2
    nmax = self.nmax

    self.sxy = zeros((nmax, 2), npfloat)
    self.vec = zeros((nmax, 2), npfloat)
    self.dist = zeros(nmax, npfloat)
    self.sv = zeros(nmax, npint)
    self.tmp = zeros(nmax, npfloat)
    self.zone = zeros(nmax, npint)

    zone_map_size = self.nz2*64
    self.zone_node = zeros(zone_map_size, npint)

    self.zone_num = zeros(self.nz2, npint)

  def __cuda_init(self):

    import pycuda.autoinit
    from helpers import load_kernel

    self.cuda_agg_count = load_kernel(
      'modules/cuda/agg_count.cu',
      'agg_count',
      subs = {'_THREADS_': self.threads}
    )

    self.cuda_agg = load_kernel(
      'modules/cuda/agg.cu',
      'agg',
      subs = {'_THREADS_': self.threads}
    )

    self.cuda_nn = load_kernel(
      'modules/cuda/nn.cu',
      'NN',
      subs = {'_THREADS_': self.threads}
    )

    self.cuda_growth = load_kernel(
      'modules/cuda/growth.cu',
      'Growth',
      subs = {'_THREADS_': self.threads}
    )

  def __init_sources(self, init_num):

    from dddUtils.random import darts_rect

    sources = darts_rect(
      init_num,
      0.5,
      0.5,
      0.9,
      0.9,
      self.sources_rad
    )

    snum = len(sources)
    self.sxy[:snum,:] = sources[:,:]
    self.snum = snum

  def __init_veins(self, veins):

    self.vnum = len(veins)
    self.vxy = veins.astype(npfloat)

  def __make_zonemap(self):

    from pycuda.driver import In
    from pycuda.driver import Out
    from pycuda.driver import InOut

    vxy = self.vxy
    vnum = self.vnum

    zone_num = self.zone_num
    zone_node = self.zone_node
    zone = self.zone

    zone_num[:] = 0

    self.cuda_agg_count(
      npint(vnum),
      npint(self.nz),
      In(vxy[:vnum,:]),
      InOut(zone_num),
      block=(self.threads,1,1),
      grid=(vnum//self.threads + 1,1)
    )

    zone_leap = zone_num[:].max()
    zone_map_size = self.nz2*zone_leap

    if zone_map_size>len(zone_node):
      print('resize, new zone leap: ', zone_map_size*2./self.nz2)
      zone_node = zeros(zone_map_size*2, npint)

    zone_node[:] = 0
    zone_num[:] = 0

    self.cuda_agg(
      npint(vnum),
      npint(self.nz),
      npint(zone_leap),
      In(vxy[:vnum,:]),
      InOut(zone_num),
      InOut(zone_node),
      Out(zone[:vnum]),
      block=(self.threads,1,1),
      grid=(vnum//self.threads + 1,1)
    )

    return zone_leap, zone_node, zone_num

  def __nn_query(self, zone_leap, zone_node, zone_num):

    from pycuda.driver import In
    from pycuda.driver import Out

    snum = self.snum
    vnum = self.vnum

    sv = self.sv[:snum]
    tmp = self.tmp[:snum]

    self.cuda_nn(
      npint(self.nz),
      npfloat(self.area_rad),
      npint(zone_leap),
      In(zone_num),
      In(self.zone_node),
      npint(snum),
      npint(vnum),
      In(self.sxy[:snum,:]),
      In(self.vxy[:vnum,:]),
      Out(sv),
      Out(tmp),
      block=(self.threads,1,1),
      grid=(snum//self.threads + 1,1)
    )

    return sv, tmp

  def __get_vs(self, sv):

    ## TODO: write as kernel?

    from collections import defaultdict
    from numpy import concatenate
    from numpy import cumsum

    vs_dict = defaultdict(list)
    vs_counts = zeros(self.vnum, npint)

    for s,v in enumerate(sv):
      if s<0 or v<0:
        continue
      vs_dict[v].append(s)
      vs_counts[v] += 1

    vs_map = concatenate([v for v in vs_dict.values()]).astype(npint)
    vs_ind = cumsum(concatenate([[0],vs_counts])).astype(npint)

    # print(vs_map)
    # print(vs_ind)
    # print(vs_counts)

    return vs_dict, vs_map, vs_ind, vs_counts

  def __growth(self, vs_map, vs_ind, vs_counts):

    from pycuda.driver import In
    from pycuda.driver import InOut

    vnum = self.vnum
    snum = self.snum
    vec = self.vec[:vnum,:]
    dst = self.tmp[:vnum]

    sxy = self.sxy[:snum, :]
    vxy = self.vxy[:vnum, :]

    dst[:] = -1.0
    vec[:,:] = -99.0

    self.cuda_growth(
      In(vs_map),
      In(vs_ind),
      In(vs_counts),
      In(sxy),
      In(vxy),
      npint(vnum),
      InOut(vec),
      InOut(dst),
      block=(self.threads,1,1),
      grid=(vnum//self.threads + 1,1)
    )

    return vec

  def step(self, t=None):

    self.itt += 1

    zone_leap, zone_node, zone_num = self.__make_zonemap()
    sv, dst = self.__nn_query(zone_leap, zone_node, zone_num)
    _, vs_map, vs_ind, vs_counts = self.__get_vs(sv)

    vec = self.__growth(vs_map, vs_ind, vs_counts)
    print(vec)

    return zone_leap, zone_node, zone_num, sv, dst

