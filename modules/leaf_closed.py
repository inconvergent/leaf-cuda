# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division


from numpy import pi
from numpy import zeros

from numpy import float32 as npfloat
from numpy import int32 as npint


TWOPI = pi*2
PI = pi



class LeafClosed(object):

  def __init__(
      self,
      size,
      stp,
      init_sources,
      init_veins,
      area_rad,
      kill_rad,
      sources_rad,
      threads = 256,
      nmax = 10000000
    ):

    self.itt = 0

    self.area_rad = area_rad
    self.kill_rad = kill_rad
    self.threads = threads
    self.nmax = nmax
    self.size = size

    self.one = 1.0/size
    self.stp = stp

    self.sources_rad = sources_rad

    # TODO: configurable
    self.max_descendants = 3

    self.__init()
    self.__init_sources(init_sources)
    self.__init_veins(init_veins)
    self.__cuda_init()

    self.sv_leap = 5*int(self.snum/self.nz2)

  def __init(self):

    self.vnum = 0
    self.snum = 0

    nz = int(1.0/(2*self.area_rad))
    self.nz = nz

    self.nz2 = self.nz**2
    nmax = self.nmax

    self.sxy = zeros((nmax, 2), npfloat)
    self.vxy = zeros((nmax, 2), npfloat)
    self.vec = zeros((nmax, 2), npfloat)

    self.sv = zeros(nmax, npint)
    self.sv_num = zeros(nmax, npint)

    self.num_descendants = zeros(nmax, npint)

    self.dst = zeros(nmax, npfloat)

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

    self.cuda_rnn = load_kernel(
      'modules/cuda/rnn.cu',
      'RNN',
      subs = {'_THREADS_': self.threads}
    )

    self.cuda_growth = load_kernel(
      'modules/cuda/growth.cu',
      'Growth',
      subs = {'_THREADS_': self.threads}
    )

  def __init_sources(self, init_sources):

    snum = len(init_sources)
    self.sxy[:snum,:] = init_sources.astype(npfloat)
    self.snum = snum

  def __init_veins(self, init_veins):

    vnum = len(init_veins)
    self.vxy[:vnum,:] = init_veins.astype(npfloat)
    self.vnum = vnum

  def __make_zonemap(self):

    from pycuda.driver import In
    from pycuda.driver import Out
    from pycuda.driver import InOut

    vxy = self.vxy
    vnum = self.vnum

    zone_num = self.zone_num
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

    if zone_map_size>len(self.zone_node):
      print('resize, new zone leap: ', zone_map_size*2./self.nz2)
      self.zone_node = zeros(zone_map_size*2, npint)

    self.zone_node[:] = 0
    zone_num[:] = 0

    self.cuda_agg(
      npint(vnum),
      npint(self.nz),
      npint(zone_leap),
      In(vxy[:vnum,:]),
      InOut(zone_num),
      InOut(self.zone_node),
      Out(zone[:vnum]),
      block=(self.threads,1,1),
      grid=(vnum//self.threads + 1,1)
    )

    return zone_leap, self.zone_node, zone_num

  def __rnn_query(self, zone_leap, zone_node, zone_num):

    from pycuda.driver import In
    from pycuda.driver import InOut
    from pycuda.driver import Out

    snum = self.snum
    vnum = self.vnum

    sv_leap = self.sv_leap
    sv = self.sv[:snum*sv_leap]
    sv_num = self.sv_num[:snum*sv_leap]
    dst = self.dst[:snum*sv_leap]

    sv_num[:] = 0
    sv[:] = -5
    dst[:] = -10.0

    self.cuda_rnn(
      npint(self.nz),
      npfloat(self.area_rad),
      npint(zone_leap),
      npint(sv_leap),
      In(zone_num),
      In(zone_node),
      npint(snum),
      npint(vnum),
      In(self.sxy[:snum,:]),
      In(self.vxy[:vnum,:]),
      InOut(sv_num),
      InOut(sv),
      InOut(dst),
      block=(self.threads,1,1),
      grid=(snum//self.threads + 1,1)
    )

    # print('sv', sv[:snum], (sv[:snum]>-1).sum())
    # print('sv_num',sv_num[:snum])
    # print('dst',dst[:snum])

    return sv_num, sv, dst

  def __get_vs(self, sv_num, sv):

    ## TODO: write as kernel?

    from collections import defaultdict
    from numpy import concatenate
    from numpy import cumsum
    from operator import itemgetter

    first = itemgetter(0)

    sv_leap = self.sv_leap
    ma = sv_num.max()
    assert  ma<sv_leap, 'sv_num exceeds sv_leap: {:d} {:d}'.format(ma, sv_leap)

    vs_dict = defaultdict(list)
    vs_num = zeros(self.vnum, npint)

    for s in xrange(self.snum):
      for k in xrange(sv_num[s]):
        v = sv[s*sv_leap+k]
        if v<0:
          continue
        vs_dict[v].append(s)
        vs_num[v] += 1

    vs_tuples = sorted(vs_dict.iteritems(), key=first)

    # for a,b in vs_tuples:
      # print(a,b)

    vs_map = concatenate([b for _,b in vs_tuples]).astype(npint)
    vs_ind = cumsum(concatenate([[0],vs_num])).astype(npint)

    return vs_map, vs_ind, vs_num

  def __growth(self, vs_map, vs_ind, vs_counts):

    from pycuda.driver import In
    from pycuda.driver import InOut

    vnum = self.vnum
    snum = self.snum
    vec = self.vec[:vnum,:]

    max_descendants = self.max_descendants
    num_descendants = self.num_descendants[:vnum]

    stp = self.stp

    sxy = self.sxy[:snum, :]
    vxy = self.vxy[:vnum, :]

    vec[:,:] = -99.0

    self.cuda_growth(
      In(vs_map),
      In(vs_ind),
      In(vs_counts),
      In(sxy),
      In(vxy),
      npint(vnum),
      InOut(vec),
      block=(self.threads,1,1),
      grid=(vnum//self.threads + 1,1)
    )

    count = 0
    warnings = 0
    abort = True
    for i in xrange(vnum):

      gv = vec[i,:]
      if gv[0]<-3.0:
        warnings += 1
        continue

      if num_descendants[i]>max_descendants:
        continue

      self.vxy[vnum+count,:] = self.vxy[i,:] + stp*gv
      num_descendants[i] += 1
      count += 1
      abort = False

    self.vnum += count

    return not abort

  def __alive_sources(self, sv_num, sv, dst):

    from numpy import ones

    sv_leap = self.sv_leap
    kill_rad = self.kill_rad

    mask = ones(self.snum)

    for s in xrange(self.snum):

      ## TODO: this can be vectorized

      near = 0
      for k in xrange(sv_num[s]):
        v = sv[s*sv_leap+k]
        if v<0:
          continue

        dd = dst[s*sv_leap+k]
        if dd<kill_rad:
          near += 1
          # mask[s] = 0
          # break

      if (sv_num[s]>0) and (near >= sv_num[s]):
        mask[s] = 0

    inds = mask.nonzero()[0]
    alive = len(inds)
    self.sxy[:alive,:] = self.sxy[inds,:]
    self.snum = alive
    return self.snum>0

  def step(self, show=None):

    while True:

      self.itt += 1

      zone_leap, zone_node, zone_num = self.__make_zonemap()
      sv_num, sv, dst = self.__rnn_query(zone_leap, zone_node, zone_num)

      # sv, sv_num is out of sync after __source_death is called
      yield sv_num, sv

      vs_map, vs_ind, vs_num = self.__get_vs(sv_num, sv)
      if not self.__growth(vs_map, vs_ind, vs_num):
        return

      if not self.__alive_sources(sv_num, sv, dst):
        return

