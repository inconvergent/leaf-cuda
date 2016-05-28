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
  float dd;
  float dx;
  float dy;

  float gx = 0.0f;
  float gy = 0.0f;

  for (int k=0; k<v_count; k++){
    ss = 2*vs_map[start + k];
    dx = sxy[ss] - vxy[vv];
    dy = sxy[ss+1] - vxy[vv+1];
    dd = sqrt(dx*dx + dy*dy);

    gx += dx/dd;
    gy += dy/dd;

  }

  dd = sqrt(gx*gx+gy*gy);
  gx /= dd;
  gy /= dd;

  vec[vv] = gx;
  vec[vv+1] = gy;
}
