#define THREADS _THREADS_

__device__ float dist(float *a, float *b, int i, int j){
    return sqrt(pow(a[2*i]-b[2*j], 2.0f)+pow(a[2*i+1]-b[2*j+1], 2.0f));
}

__global__ void NN(
  int nz,
  float rad,
  int zone_leap,
  int *zone_num,
  int *zone_node,
  int snum,
  int vnum,
  float *sxy,
  float *vxy,
  int *sv,
  float *dst
){
  const int s = blockIdx.x*THREADS + threadIdx.x;

  if (s>=snum){
    return;
  }

  const int ss = 2*s;

  const int za = (int)floor(sxy[ss]*nz);
  const int zb = (int)floor(sxy[ss+1]*nz);

  float dd = -1.0f;

  int v = -4;
  int zk;

  int r = -3;
  float mi = 99999.0f;

  int cand_count = 0;

  for (int a=max(za-1,0);a<min(za+2,nz);a++){
    for (int b=max(zb-1,0);b<min(zb+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){
        cand_count += 1;

        v = zone_node[zk*zone_leap+k];
        dd = dist(vxy, sxy, v, s);
        if (dd>rad){
          continue;
        }

        if (dd<mi){
          mi = dd;
          r = v;
        }
      }
    }
  }

  sv[s] = r;
  dst[s] = mi;
}
