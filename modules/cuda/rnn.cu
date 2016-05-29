#define THREADS _THREADS_

__device__ float dist(float *a, float *b, int ii, int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__device__ int calc_zones(int za, int zb, int nz, int *Z){
  int num = 0;
  for (int a=max(za-1,0);a<min(za+2,nz);a++){
    for (int b=max(zb-1,0);b<min(zb+2,nz);b++){
      Z[num] = a*nz+b;
      num += 1;
    }
  }
  return num;
}

__device__ bool is_relative(
  int ZN,
  int *Z,
  int zone_leap,
  int *zone_num,
  int *zone_node,
  float rad,
  float *sxy,
  float *vxy,
  int ss,
  int vv
){

  int uu;
  int z;
  float dd;

  dd = dist(sxy, vxy, ss, vv);
  if (dd>rad){
    return false;
  }

  for (int zk=0;zk<ZN;zk++){
    z = Z[zk];
    for (int k=0;k<zone_num[z];k++){
      uu = 2*zone_node[z*zone_leap+k];

      if (dd>max(dist(sxy, vxy, ss, uu), dist(vxy, vxy, vv, uu))){
        return false;
      }

    }
  }
  return true;
}

__global__ void RNN(
  int nz,
  float rad,
  int zone_leap,
  int sv_leap,
  int *zone_num,
  int *zone_node,
  int snum,
  int vnum,
  float *sxy,
  float *vxy,
  int *sv_num,
  int *sv,
  float *dst
){
  const int s = blockIdx.x*THREADS + threadIdx.x;

  if (s>=snum){
    return;
  }

  const int ss = 2*s;

  int Z[9];
  const int za = (int)floor(sxy[ss]*nz);
  const int zb = (int)floor(sxy[ss+1]*nz);
  const int ZN = calc_zones(za, zb, nz, Z);

  int v = -4;
  int vv = -8;
  int z = 33;

  bool relative;
  int count = 0;

  for (int zk=0;zk<ZN;zk++){
    z = Z[zk];
    for (int k=0;k<zone_num[z];k++){
      v = zone_node[z*zone_leap+k];
      vv = 2*v;

      // is relative neightbor?
      relative = is_relative(
        ZN,
        Z,
        zone_leap,
        zone_num,
        zone_node,
        rad,
        sxy,
        vxy,
        ss,
        vv
      );

      if (relative){
        sv[s*sv_leap+count] = v;
        dst[s*sv_leap+count] = dist(vxy, sxy, vv, ss);
        count += 1;
      }

    }
  }

  sv_num[s] = count;
}
