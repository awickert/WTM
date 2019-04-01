#ifndef _richdem_flat_resolution_parallel_hpp_
#define _richdem_flat_resolution_parallel_hpp_

#include <richdem/common/constants.hpp>
#include <omp.h>
#include <queue>
#include <deque>

#include "DisjointHashIntSet.hpp"

namespace richdem {

template<class elev_t, class flowdir_t, Topology topo>
void ResolveFlat(
  const uint64_t c0,
  const Array2D<elev_t> &dem,
  Array2D<flowdir_t>    &flowdirs
){
  //A D4 or D8 topology can be used.
  const int *dx, *dy, *dinverse;
  const double *dr;
  int neighbours;
  TopologicalResolver<topo>(dx,dy,dr,dinverse,neighbours);

  std::unordered_set<uint64_t> in_flat;
  std::deque<uint64_t> lower;
  //Find lower
  std::queue<uint64_t> q;
  q.push(c0);
  in_flat.insert(c0);
  while(!q.empty()){
    const auto ci = q.front();
    q.pop();

    int cx,cy;
    dem.iToxy(ci,cx,cy);

    bool has_lower = false;
    for(int n=1;n<neighbours;n++){
      const int nx = cx+dx[n];
      const int ny = cy+dy[n];
      if(!dem.inGrid(nx,ny))
        continue;
      const int ni = dem.xyToI(nx,ny);
      if(dem(ci)>dem(ni)){
        has_lower = true;
        break;
      } else if(dem(ci)==dem(ni) && in_flat.count(ni)==0){
        q.push(ni);
        in_flat.insert(ni);
      }
    }
    if(has_lower)
      lower.push_back(ci);
  }

  assert(q.empty());

  //If there is no way out of the flat to lower ground, suck everything
  //arbitrarily to this one cell
  if(lower.empty())
    lower.push_back(c0);

  q = std::queue<uint64_t>(std::move(lower));
  lower.clear();
  lower.shrink_to_fit();

  while(!q.empty()){
    const auto ci = q.front();
    q.pop();

    int cx,cy;
    dem.iToxy(ci,cx,cy);

    for(int n=1;n<neighbours;n++){
      const int nx=cx+dx[n];
      const int ny=cy+dy[n];
      if(!dem.inGrid(nx,ny))
        continue;
      const auto ni=dem.xyToI(nx,ny);
      if(in_flat.count(ni) && flowdirs(ni)==NO_FLOW){
        flowdirs(ni) = dinverse[n];
        q.push(ni);
      }
    }
  }
}



template<class elev_t, class flowdir_t, Topology topo>
void FlatResolutionParallel(
  const Array2D<elev_t> &dem,
  Array2D<flowdir_t>    &flowdirs
){
  //A D4 or D8 topology can be used.
  const int *dx, *dy, *dinverse;
  const double *dr;
  int neighbours;
  TopologicalResolver<topo>(dx,dy,dr,dinverse,neighbours);

  const auto FlatSetCell = [&](
    const int x,
    const int y,
    DisjointHashIntSet<int64_t> &dhis
  ){
    bool    has_lower  = false;
    int64_t nset       = -1;
    const auto ci      = dem.xyToI(x,y);
    const auto my_elev = dem(x,y);
    for(int n=1;n<=neighbours;n++){
      const int nx     = x+dx[n];
      const int ny     = y+dy[n];
      if(!dem.inGrid(nx,ny))
        continue;
      const int ni     = dem.xyToI(nx,ny);
      const auto nelev = dem(x,y);
      if(my_elev>nelev){
        has_lower = true;
        break;
      } else if(my_elev==nelev && dhis.isSet(ni)){
        nset = ni;
      }
    }

    if(has_lower)
      return;

    if(nset==-1){
      dhis.makeSet(ci);
    } else {
      dhis.unionSet(ci,nset);
    }
  };

  DisjointHashIntSet<int64_t> dhis;

  #pragma omp declare reduction (merge : DisjointHashIntSet<int64_t>: omp_out=MergeDisjointHashIntSet(omp_out, omp_in))

  int thread_count=-1;

  #pragma omp parallel default(none) shared(dem) reduction(merge:dhis) reduction(max:thread_count)
  {
    thread_count = omp_get_num_threads(); //Will be the same for all threads
    //Calculate horizontal stripes of the dataset to hand to each thread
    const int tnum   = omp_get_thread_num();
    const int inc    = dem.height()/thread_count;
    int lbound = tnum*inc;
    int ubound = std::min((tnum+1)*inc,dem.height());
    //But leave a seam between each stripe that will be calculated in serial
    if(lbound>0)
      lbound++;
    if(ubound<dem.height())
      ubound--;

    for(int y=lbound;y<ubound;y++)
    for(int x=0;x<dem.width();x++)
      FlatSetCell(x,y,dhis);
  }

  //Stitch seams together
  for(int tnum=1;tnum<thread_count;tnum++){
    const int inc = dem.height()/thread_count;
    int lbound    = tnum*inc-1;
    for(int y=lbound;y<lbound+2;y++)
    for(int x=0;x<dem.width();x++)
      FlatSetCell(x,y,dhis);
  }

  //Identify unique flats
  std::vector<int64_t> uflats;
  for(const auto &kv: dhis.getParentData())
    uflats.push_back(dhis.findSet(kv.second));
  auto last = std::unique(uflats.begin(), uflats.end());
  uflats.erase(last, uflats.end()); 

  dhis.clear();
  dhis.shrink_to_fit();

  //For each flat, resolve it
  #pragma omp parallel for schedule(dynamic)
  for(unsigned int i=0;i<uflats.size();i++)
    ResolveFlat<elev_t,flowdir_t,topo>(uflats[i],dem,flowdirs);
}

}

#endif
