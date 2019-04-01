#ifndef _richdem_flat_resolution_parallel_hpp_
#define _richdem_flat_resolution_parallel_hpp_

#include <iostream>
#include <richdem/common/Array2D.hpp>
#include <richdem/common/constants.hpp>
#include <deque>
#include <limits>
#include <omp.h>
#include <queue>

#include "DisjointHashIntSet.hpp"

namespace richdem {

template<class elev_t, class flowdir_t, Topology topo>
void ResolveFlat(
  const int64_t c0,
  const DisjointHashIntSet<int64_t> &dhis,
  const Array2D<elev_t> &dem,
  Array2D<flowdir_t>    &flowdirs
){
  // #pragma omp critical
  // std::cout<<"Found flat at "<<c0<<std::endl;
  //A D4 or D8 topology can be used.
  const int *dx, *dy, *dinverse;
  const double *dr;
  int neighbours;
  TopologicalResolver<topo>(dx,dy,dr,dinverse,neighbours);

  //Use a breadth-first search to identify all those cells on the edge of the
  //flat
  std::unordered_set<uint64_t> visited; //Cells we've visited (for ensuring BFS progresses)
  std::deque<uint64_t> lower;           //Cells which have lower neighbours (edge of flat)
  std::queue<uint64_t> q;               //Frontier of the BFS
  q.push(c0);
  visited.insert(c0);
  while(!q.empty()){
    const auto ci = q.front();          //Get next cell
    q.pop();

    int cx,cy;                          //Coordinates of next cell
    dem.iToxy(ci,cx,cy);
    const auto my_elev = dem(ci);

    bool has_lower = false;             //Does it have a lower neighbour?
    for(int n=1;n<neighbours;n++){
      const int  nx      = cx+dx[n];    //Get neighbour coordinates
      const int  ny      = cy+dy[n];
      const auto in_grid = dem.inGrid(nx,ny); //Is neighbour in grid?
      int ni             = -1;          //Temporary index of neigbour
      if(in_grid)                       //If neighbour is in grid
        ni = dem.xyToI(nx,ny);          //Get its index
      if(!in_grid || my_elev>dem(ni)) { //If neighbour is not in grid or lower than me
        has_lower = true;               //Then I have a lower neighbour
      } else if(dhis.isSet(ni) && dhis.findSet(ni)==c0 && visited.count(ni)==0){ //If neighbour is in my flat and hasn't been visited
        q.push(ni);                     //Prepare to visit neighbour
        visited.insert(ni);
      }
      // std::cerr<<"ni="<<ni<<" is_set="<<dhis.isSet(ni);
      // if(dhis.isSet(ni)) std::cerr<<" findset="<<dhis.findSet(ni);
      // std::cerr<<std::endl;
    }
    if(has_lower)
      lower.push_back(ci);
  }

  assert(q.empty());

  //If there is no way out of the flat to lower ground, suck everything
  //arbitrarily to this one cell
  if(lower.empty())
    lower.push_back(c0);

  // std::cerr<<"Lower: ";
  // for(const auto &x: lower)
  //   std::cerr<<x<<" ";
  // std::cerr<<std::endl;

  // std::cerr<<"Visited:\n";
  // for(int y=0;y<dem.height();y++){
  //   for(int x=0;x<dem.width();x++)
  //     std::cerr<<(int)visited.count(dem.xyToI(x,y));
  //   std::cerr<<std::endl;
  // }

  //Let's visit all of the cells in the flat using a breadth-first traversal,
  //starting at the lower edges
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
      if(visited.count(ni) && flowdirs(ni)==NO_FLOW && ni!=c0){
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

  //Assign steepest-slope flow directions to all cells that don't already have
  //them
  #pragma omp parallel for default(none) shared(dem,flowdirs,dx,dy,dr,neighbours) collapse(2)
  for(int y=0;y<dem.height();y++)
  for(int x=0;x<dem.width ();x++){
    if(flowdirs(x,y)!=NO_FLOW)
      continue;
    double greatest_slope = 0;
    auto   greatest_n    = NO_FLOW;
    const auto my_elev = dem(x,y);
    for(int n=1;n<=neighbours;n++){
      const int nx = x+dx[n];
      const int ny = y+dy[n];
      if(!dem.inGrid(nx,ny)){
        greatest_slope = -std::numeric_limits<double>::infinity();
        greatest_n     = n;
        break;
      }
      double slope = (my_elev-dem(nx,ny))/dr[n];
      if(slope>greatest_slope){
        greatest_slope = slope;
        greatest_n     = n;
      }
    }
    flowdirs(x,y) = greatest_n;
  }

  // std::cerr<<"Flowdirs:";
  // flowdirs.printAll();

  DisjointHashIntSet<int64_t> dhis;


  //Identifies whether a cell is in a flat and, if so, makes a note of this and
  //merges the cell into any preexisting flats its neighbours at the same
  //elevation are at. Even if the cell is not part of a flat, any neighbours of
  //the same elevation which are in flats are merged into this cell. In this
  //way, we find the drainage zones of the flat.
  #pragma omp declare reduction (merge : DisjointHashIntSet<int64_t>: omp_out=MergeDisjointHashIntSet(omp_out, omp_in))
  #pragma omp parallel for default(none) shared(dem,flowdirs,dx,dy,neighbours) reduction(merge:dhis)
  for(int y=0;y<dem.height();y++)
  for(int x=0;x<dem.width();x++){
    const auto ci         = dem.xyToI(x,y);  //Flat index of cell
    const auto my_elev    = dem(ci);         //Elevation of cell
    const auto my_flowdir = flowdirs(ci);
    for(int n=1;n<=neighbours;n++){          //Consider neighbours
      const int nx = x+dx[n];                //Get x-coordinate of neighbour
      const int ny = y+dy[n];                //Get y-coordinate of neighbour
      if(!dem.inGrid(nx,ny))
        continue;
      const int  ni    = dem.xyToI(nx,ny);   //Id of neighbour
      const auto nelev = dem(ni);            //Neighbour elevation
      if(my_elev==nelev && (flowdirs(ni)==NO_FLOW || my_flowdir==NO_FLOW)){
        dhis.unionSet(ci,ni);
      }
    }
  }

  // std::cerr<<"Sets:\n";
  // for(int y=0;y<dem.height();y++){
  //   for(int x=0;x<dem.width();x++){
  //     if(dhis.isSet(dem.xyToI(x,y)))
  //       std::cerr<<std::setw(3)<<dhis.findSet(dem.xyToI(x,y))<<" ";
  //     else
  //       std::cerr<<std::setw(3)<<"-"<<" ";
  //   }
  //   std::cerr<<std::endl;
  // }

  //Identify unique flats. If we don't do this, than more than one process might
  //begin filling in a flat at once! This takes O(N) time in the number of cells
  //in flats, but could be parallelized with an appropriate hash table (C++ hash
  //table doesn't allow random access to entries) and benign race conditions
  //arising from finding parents in the disjoint-set.
  std::unordered_set<int64_t> uflats;
  for(const auto &kv: dhis.getParentData())
    uflats.insert(dhis.findSet(kv.second));

  //Transfor into a vector so we can parallelize flat resolution
  std::vector<int64_t> vuflats(uflats.begin(),uflats.end());

  //For each flat, resolve it
  #pragma omp parallel for schedule(dynamic)
  for(unsigned int i=0;i<vuflats.size();i++)
    ResolveFlat<elev_t,flowdir_t,topo>(vuflats[i],dhis,dem,flowdirs);
}

}

#endif
