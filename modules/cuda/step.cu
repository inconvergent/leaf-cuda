#define THREADS _THREADS_

__global__ void step(
  int n,
  int nz,
  int zone_leap,
  float *xy,
  int *zone_num,
  int *zone_node,
  float stp
){
  const int i = blockIdx.x*THREADS + threadIdx.x;

  if (i>=n){
    return;
  }

  const int ii = 2*i;

  const int zi = (int)floor(xy[ii]*nz);
  const int zj = (int)floor(xy[ii+1]*nz);

  float dx = 0.0f;
  float dy = 0.0f;
  float dd = 0.0f;


  int j;
  int jj;
  int zk;

  int cand_count = 0;
  int total_count = 0;

  // unlinked
  for (int a=max(zi-1,0);a<min(zi+2,nz);a++){
    for (int b=max(zj-1,0);b<min(zj+2,nz);b++){
      zk = a*nz+b;
      for (int k=0;k<zone_num[zk];k++){

        j = zone_node[zk*zone_leap+k];

        if (i==j){
          continue;
        }

        /*jj = 2*j;*/
        /*total_count += 1;*/
        /*dx = xy[ii] - xy[jj];*/
        /*dy = xy[ii+1] - xy[jj+1];*/
        /*dd = sqrt(dx*dx+dy*dy);*/

      }
    }
  }

  // persist
  /*dxy[ii] = sx*stp;*/
  /*dxy[ii+1] = sy*stp;*/

}
