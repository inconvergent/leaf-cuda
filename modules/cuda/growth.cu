#define THREADS _THREADS_

__global__ void Growth(
  int *vs_map,
  int *vs_ind,
  int *vs_counts,
  float *sxy,
  float *vxy,
  int vnum,
  float *vec
){
  const int v = blockIdx.x*THREADS + threadIdx.x;
  const int vv = 2*v;

  if (v>=vnum){
    return;
  }

  const int v_count = vs_counts[v];
  const int start = vs_ind[v];

  if (v_count<1){
    vec[vv] = -100.0f;
    vec[vv+1] = -100.0f;
    return;
  }

  int ss;
  float gx;
  float gy;
  float dd;

  float mx = 0.0f;
  float my = 0.0f;
  int count = 0;

  for (int k=0; k<v_count; k++){
    ss = 2*vs_map[start + k];
    mx += sxy[ss];
    my += sxy[ss+1];
    count += 1;
  }

  gx = mx/(float)count-vxy[vv];
  gy = my/(float)count-vxy[vv+1];
  dd = sqrt(gx*gx+gy*gy);

  vec[vv] = gx/dd;
  vec[vv+1] = gy/dd;
}
