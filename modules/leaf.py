# -*- coding: utf-8 -*-





from numpy import pi
from numpy import zeros

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import bool as npbool

from operator import itemgetter
first = itemgetter(0)


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

    self.vnum = 0
    self.enum = 0

    nz = int(1.0/(2*self.area_rad))
    self.nz = nz

    self.nz2 = self.nz**2
    nmax = self.nmax

    self.vec = zeros((nmax,2), npfloat)
    self.edges = zeros((nmax,2), npint)
    self.parent = zeros(nmax, npint)
    self.parent[:] = -1

    self.zone = zeros(nmax, npint)

    zone_map_size = self.nz2*64
    self.zone_node = zeros(zone_map_size, npint)
    self.has_descendants = zeros(nmax, npbool)
    self.gen = zeros(nmax, npint)

    self.zone_num = zeros(self.nz2, npint)

    self.__init_sources(init_sources)
    self.__init_veins(init_veins)
    self.__cuda_init()

    sv_leap = 30*int(self.snum/self.nz2)
    self.sv_leap = sv_leap

    sv_size = sv_leap*self.snum
    self.sv_size = sv_size

    self.sv = zeros(sv_size, npint)
    self.sv_num = zeros(sv_size, npint)
    self.dst = zeros(sv_size, npfloat)

  def width_calc(self, scale, min_width, po=1.0):

    from numpy import ones
    from numpy import power

    vnum = self.vnum
    width = ones(vnum, 'float')
    parent = self.parent

    for v in range(vnum):

      n = parent[v]
      if n<0:
        continue
      while True:
        if n<0:
          break
        width[n] += 1.0
        n = parent[n]

    width[:] = power(width, po)
    ma = width.max()
    width /= ma
    width *= scale
    width[width<min_width] = min_width

    return width

  def __cuda_init(self):

    import pycuda.autoinit
    from .helpers import load_kernel

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

    from numpy import ones

    snum = len(init_sources)
    self.snum = snum
    self.sxy = init_sources.astype(npfloat)
    self.smask = ones(snum, npbool)

  def __init_veins(self, init_veins):

    self.vxy = zeros((self.nmax,2), npfloat)
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

    snum = self.snum
    vnum = self.vnum

    sv_size = self.sv_size

    sv = self.sv[:sv_size]
    sv_num = self.sv_num[:sv_size]
    dst = self.dst[:sv_size]

    sv_num[:] = 0
    sv[:] = -5
    dst[:] = -10.0

    self.cuda_rnn(
      npint(self.nz),
      npfloat(self.area_rad),
      npfloat(self.kill_rad),
      npint(zone_leap),
      npint(self.sv_leap),
      In(zone_num),
      In(zone_node),
      npint(snum),
      npint(vnum),
      In(self.smask),
      In(self.sxy),
      In(self.vxy[:vnum,:]),
      InOut(sv_num),
      InOut(sv),
      InOut(dst),
      block=(self.threads,1,1),
      grid=(snum//self.threads + 1,1)
    )

    return sv_num, sv, dst

  def __get_vs(self, sv_num, sv):

    ## TODO: write as kernel?
    from collections import defaultdict
    from numpy import concatenate
    from numpy import cumsum

    sv_leap = self.sv_leap
    ma = sv_num[:self.snum*sv_leap].max()
    assert  ma<sv_leap, 'sv_num exceeds sv_leap: {:d} {:d}'.format(ma, sv_leap)

    vs_dict = defaultdict(list)
    vs_num = zeros(self.vnum, npint)

    for s in range(self.snum):
      for k in range(sv_num[s]):
        v = sv[s*sv_leap+k]
        if v<0:
          continue
        vs_dict[v].append(s)
        vs_num[v] += 1

    vs_tuples = sorted(iter(vs_dict.items()), key=first)

    vs_map = concatenate([b for _,b in vs_tuples]).astype(npint)
    vs_ind = cumsum(concatenate([[0],vs_num])).astype(npint)

    return vs_dict, vs_map, vs_ind, vs_num

  def __get_vs_xy(self, vs_dict):

    vxy = self.vxy
    sxy = self.sxy

    res = []

    for v,ss in vs_dict.items():
      for s in ss:
        res.append((list(vxy[v,:]), list(sxy[s,:])))

    return res

  def __remove_obsolete_sources(self):

    from collections import defaultdict

    obsolete_sources = defaultdict(list)
    sv_leap = self.sv_leap
    kill_rad = self.kill_rad
    vxy = self.vxy
    has_descendants = self.has_descendants
    gen = self.gen
    edges = self.edges
    parent = self.parent
    enum = self.enum
    sxy = self.sxy
    vnum = self.vnum

    zone_leap, zone_node, zone_num = self.__make_zonemap()
    sv_num, sv, dst = self.__rnn_query(zone_leap, zone_node, zone_num)

    for s in range(self.snum):
      near = 0
      vv = []
      for k in range(sv_num[s]):
        v = sv[s*sv_leap+k]
        if v<0:
          continue
        vv.append(v)

        dd = dst[s*sv_leap+k]
        if dd<kill_rad:
          near += 1

      if (sv_num[s]>0) and (near >= sv_num[s]):
        obsolete_sources[s] = set(vv)

    die = list(obsolete_sources.keys())
    self.smask[die] = False

    for s,vv in obsolete_sources.items():
      vvv = list(vv)
      if len(vvv)>1:
        vxy[vnum,:] = sxy[s,:]
        parent[vnum] = min(vvv)
        gen[vnum] = gen[vvv].max()
        has_descendants[vvv] = True
        for v in vvv:
          edges[enum, :] = [v,vnum]
          enum += 1
        vnum += 1

    self.enum = enum
    self.vnum = vnum

  def __growth(
    self,
    zone_leap,
    zone_num,
    zone_node,
    vs_map,
    vs_ind,
    vs_counts
  ):

    from pycuda.driver import In
    from pycuda.driver import InOut

    vnum = self.vnum
    enum = self.enum
    has_descendants = self.has_descendants
    gen = self.gen

    vec = self.vec[:vnum,:]

    stp = self.stp
    kill_rad = self.kill_rad

    edges = self.edges
    parent = self.parent
    sxy = self.sxy
    vxy = self.vxy

    vec[:,:] = -99.0

    self.cuda_growth(
      npint(self.nz),
      npfloat(kill_rad),
      npfloat(stp),
      npint(zone_leap),
      In(zone_num),
      In(zone_node),
      In(vs_map),
      In(vs_ind),
      In(vs_counts),
      In(sxy),
      In(vxy[:vnum,:]),
      npint(vnum),
      InOut(vec),
      block=(self.threads,1,1),
      grid=(vnum//self.threads + 1,1)
    )

    abort = True
    for i in range(vnum):
      gv = vec[i,:]
      if gv[0]<-3.0:
        continue

      if has_descendants[i]:
        gen[vnum] = gen[i]+1
      else:
        gen[vnum] = gen[i]
      has_descendants[i] = True
      edges[enum, :] = [i,vnum]
      parent[vnum] = i
      vxy[vnum,:] = gv
      abort = False
      enum += 1
      vnum += 1

    self.enum = enum
    self.vnum = vnum

    return abort

  def step(self, show=None):

    while True:

      self.itt += 1

      zone_leap, zone_node, zone_num = self.__make_zonemap()
      sv_num, sv, dst = self.__rnn_query(zone_leap, zone_node, zone_num)

      vs_dict, vs_map, vs_ind, vs_num = self.__get_vs(sv_num, sv)

      yield self.__get_vs_xy(vs_dict)

      abort = self.__growth(
        zone_leap,
        zone_num,
        zone_node,
        vs_map,
        vs_ind,
        vs_num
      )

      if abort:
        return

      self.__remove_obsolete_sources()

