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
      init_sources,
      nz = 32,
      threads = 256,
      nmax = 1000000
    ):

    self.itt = 0

    self.nz = nz
    self.threads = threads
    self.nmax = nmax
    self.size = size

    self.one = 1.0/size
    self.stp = stp
    self.init_sources = init_sources

    self.sources_dst = self.one*5.0

    self.__init()
    self.__init_sources(self.init_sources)
    self.__cuda_init()

  def __init(self):

    self.vnum = 0
    self.snum = 0

    self.nz2 = self.nz**2
    nmax = self.nmax

    self.sxy = zeros((nmax, 2), npfloat)
    self.xy = zeros((nmax, 2), npfloat)
    self.dxy = zeros((nmax, 2), npfloat)
    self.tmp = zeros((nmax, 1), npfloat)
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
    self.cuda_step = load_kernel(
      'modules/cuda/step.cu',
      'step',
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
      self.sources_dst
    )

    n = len(sources)
    self.sxy[:n,:] = sources[:,:]
    self.snum = n

  def step(self, t=None):

    import pycuda.driver as drv

    self.itt += 1

    snum = self.snum
    sxy = self.sxy
    # dxy = self.dxy
    # tmp = self.tmp
    blocks = snum//self.threads + 1

    self.zone_num[:] = 0

    self.cuda_agg_count(
      npint(snum),
      npint(self.nz),
      drv.In(sxy[:snum,:]),
      drv.InOut(self.zone_num),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

    zone_leap = self.zone_num[:].max()
    zone_map_size = self.nz2*zone_leap

    if zone_map_size>len(self.zone_node):
      print('resize, new zone leap: ', zone_map_size*2./self.nz2)
      self.zone_node = zeros(zone_map_size*2, npint)

    self.zone_num[:] = 0

    self.cuda_agg(
      npint(snum),
      npint(self.nz),
      npint(zone_leap),
      drv.In(sxy[:snum,:]),
      drv.InOut(self.zone_num),
      drv.Out(self.zone_node),
      drv.Out(self.zone[:snum]),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

    self.cuda_step(
      npint(snum),
      npint(self.nz),
      npint(zone_leap),
      drv.In(sxy[:snum,:]),
      drv.In(self.zone_num),
      drv.In(self.zone_node),
      npfloat(self.stp),
      block=(self.threads,1,1),
      grid=(blocks,1)
    )

    # xy[:num,:] += dxy[:num,:]

