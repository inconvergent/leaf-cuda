#define THREADS _THREADS_

__device__ float dist(const float *a, const float *b, const int ii, const int jj){
    return sqrt(powf(a[ii]-b[jj], 2.0f)+powf(a[ii+1]-b[jj+1], 2.0f));
}

__global__ void Growth(
  const int nz,
  const float kill_rad,
  const float stp,
  const int zone_leap,
  const int *zone_num,
  const int *zone_node,
  const int *vs_map,
  const int *vs_ind,
  const int *vs_counts,
  const float *sxy,
  const float *vxy,
  const int vnum,
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
  float gx = 0.0f;
  float gy = 0.0f;

  float dx = 0.0f;
  float dy = 0.0f;
  float dd = 0.0f;

  for (int k=0; k<v_count; k++){
    ss = 2*vs_map[start + k];
    dx = sxy[ss]-vxy[vv];
    dy = sxy[ss+1]-vxy[vv+1];
    dd = sqrt(dx*dx+dy*dy);

    /*if (dd<kill_rad){*/
      /*vec[vv] = -100.0f;*/
      /*vec[vv+1] = -100.0f;*/
      /*return;*/
    /*}*/
    gx += dx/dd;
    gy += dy/dd;
  }

  dd = sqrt(gx*gx+gy*gy);
  gx /= dd;
  gy /= dd;

  // test if new node collides with previous nodes
  int cand;
  const float tx = vxy[vv]+gx*stp;
  const float ty = vxy[vv+1]+gy*stp;
  const int za = (int)floor(tx*nz);
  const int zb = (int)floor(ty*nz);
  int z;
  float mi = 99999.0f;

  for (int a=max(za-1,0);a<min(za+2,nz);a++){
    for (int b=max(zb-1,0);b<min(zb+2,nz);b++){
      z = a*nz+b;
      for (int k=0;k<zone_num[z];k++){

        cand = 2*zone_node[z*zone_leap+k];

        if (cand==vv){
          continue;
        }

        dd = sqrt(powf(vxy[cand]-tx,2.0f)+
                  powf(vxy[cand+1]-ty,2.0f));

        if (dd<mi){
          mi = dd;
        }
      }
    }
  }

  if (mi>stp){
    vec[vv] = tx;
    vec[vv+1] = ty;
  }
  else{
    vec[vv] = -100.0f;
    vec[vv+1] = -100.0f;
  }
}
