#define THREADS _THREADS_

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
  int zk;
  float dd;
  float dx;
  float dy;

  for (int z=0;z<ZN;z++){
    zk = Z[z];
    for (int k=0;k<zone_num[zk];k++){
      uu = 2*zone_node[zk*zone_leap+k];

      dx = sxy[ss]-vxy[vv];
      dy = sxy[ss+1]-vxy[vv+1];
      dd = sqrt(dx*dx+dy*dy);

      if (dd>rad){
        return false;
      }

      if (dd>max(
          sqrt(powf(sxy[ss] - vxy[vv],2.0f) + powf(sxy[ss+1] - vxy[vv+1],2.0f)),
          sqrt(powf(vxy[uu] - vxy[vv],2.0f) + powf(vxy[uu+1] - vxy[vv+1],2.0f))
        )
      ){
        return false;
      }
    }
  }
  return true;
}

__device__ float dist(float *a, float *b, int i, int j){
    return sqrt(pow(a[2*i]-b[2*j], 2.0f)+pow(a[2*i+1]-b[2*j+1], 2.0f));
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

  int v;
  int zk;

  bool relative;

  int count = 0;

  for (int z=0;z<ZN;z++){
    zk = Z[z];
    for (int k=0;k<zone_num[zk];k++){
      v = zone_node[zk*zone_leap+k];
      relative = is_relative(ZN, Z, zone_leap, zone_num, zone_node, rad, sxy, vxy, 2*s, 2*v);

      if (relative){

        sv[s*sv_leap+count] = v;
        dst[s*sv_leap+count] = dist(vxy, sxy, v, s);
        count += 1;
      }
    }
  }

  sv_num[s] = count;
}