#define THREADS _THREADS_

__device__ float dist(float *a, float *b, int i, int j){
    return sqrt(pow(a[2*i]-b[2*j], 2.0f)+pow(a[2*i+1]-b[2*j+1], 2.0f));
}

__global__ void NN(
  int nz,
  int zone_leap,
  int *zone_num,
  int *zone_node,
  int snum,
  float *sxy,
  int vnum,
  float *vxy,
  int *res,
  float *tmp,
  float stp
){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i>=vnum){
    return;
  }

  const int ii = 2*i;

  const int za = (int)floor(vxy[ii]*nz);
  const int zb = (int)floor(vxy[ii+1]*nz);

  float dd;

  int j;
  int zk;

  int r = -1;
  float mindst = 100000.0f;

  for (int a=max(za-1,0);a<min(za+2,nz);a++){
    for (int b=max(zb-1,0);b<min(zb+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){

        j = zone_node[zk*zone_leap+k];

        dd = dist(vxy, sxy, i, j);

        if (dd<mindst){
          mindst = dd;
          r = j;
        }
      }
    }
  }

  res[i] = r;
  tmp[i] = mindst;

}
