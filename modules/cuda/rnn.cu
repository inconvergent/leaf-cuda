#define THREADS _THREADS_

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__device__ int calc_zones(const int za, const int zb, const int nz, int *Z){
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
  const int ZN,
  const int *Z,
  const int zone_leap,
  const int *zone_num,
  const int *zone_node,
  const float area_rad,
  const float kill_rad,
  const float *sxy,
  const float *vxy,
  const int ss,
  const int vv
){

  int uu;
  int z;
  float su;
  float vu;

  float dd = dist(sxy, vxy, ss, vv);


  if (dd>area_rad){
    return false;
  }

  for (int zk=0;zk<ZN;zk++){
    z = Z[zk];
    for (int k=0;k<zone_num[z];k++){

      uu = 2*zone_node[z*zone_leap+k];

      if (vv == uu){
        continue;
      }

      su = dist(sxy, vxy, ss, uu);
      vu = dist(vxy, vxy, vv, uu);

      if (dd>max(su, vu)){
        return false;
      }

    }
  }
  return true;
}

__global__ void RNN(
  const int nz,
  const float area_rad,
  const float kill_rad,
  const int zone_leap,
  const int sv_leap,
  const int *zone_num,
  const int *zone_node,
  const int snum,
  const int vnum,
  const bool *smask,
  const float *sxy,
  const float *vxy,
  int *sv_num,
  int *sv,
  float *dst
){
  const int s = blockIdx.x*THREADS + threadIdx.x;

  if (s>=snum){
    return;
  }

  if (!smask[s]){
    return;
  }

  const int ss = 2*s;

  int Z[9];
  const int za = (int)floor(sxy[ss]*nz);
  const int zb = (int)floor(sxy[ss+1]*nz);
  const int ZN = calc_zones(za, zb, nz, Z);

  int v = -4;
  int vv = -8;
  int z;

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
        area_rad,
        kill_rad,
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
