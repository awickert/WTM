//DisjointHashIntSet
//Author: Richard Barnes (rijard.barnes@gmail.com)
#ifndef _disjoint_hash_int_sets_
#define _disjoint_hash_int_sets_

#include <unordered_map>

//A disjoint-set/union-find class. Starting from a collection of sets, this data
//structure efficiently keeps track of which sets have been merged. A hash table
//is employed so that any integer type may be used as a set key. The data
//structure therefore takes O(K) space where K is the number of unique keys. If
//only `findSet()` and `unionSet()` are used, then all accesses are in O(a(N))
//time, where `a()` is the inverse Ackermann function. For all practical
//purposes, this is `(1). If `mergeAintoB()` is used then `findSet()` can have a
//worst- case of `O(N)`.
template<class T>
class DisjointHashIntSet {
 private:
  //How many sets are children of this set. Initially 0.
  std::unordered_map<T, uint32_t> rank;    
  //Which set is this set's parent. May be the set itself.
  std::unordered_map<T, T> parent;

  template<class U> friend DisjointHashIntSet<U>& MergeDisjointHashIntSet(DisjointHashIntSet<U> &out, DisjointHashIntSet<U> &in);

  //Follows a set's chain of parents until a set which is its own parent is
  //reached. This ultimate parent's id is returned as the representative id of
  //the set in question. Note that this collapses the chain of parents so that
  //after this method has run every set between the one in question and the
  //ultimate parent points to the ultimate parent. This means that while the
  //first call to this function may take `O(N)` lookups in the worst-case (less
  //due to the use of ranks, as explained below), subsequent calls to any set in
  //the chain will take `O(1)` time. This technique is known as "path
  //compression".
  T _findSet(const T n){
    if(parent[n]==n){                  //Am I my own parent?
      return n;                        //Yes: I represent the set in question.
    } else {                           //No.
      //Who is my parent's ultimate parent? Make them my parent.
      return parent[n] = findSet(parent[n]); 
    }
  }

  T _findSet(const T n) const {
    if(parent.at(n)==n)
      return n;
    else
      return findSet(parent.at(n));
  }

 public:
  //Construct a DisjointHashIntSet without any sets. Sets will be dynamically
  //created as the data structure is used.
  DisjointHashIntSet(){
    rank.reserve(1000);
    parent.reserve(1000);
  }

  //Create a DisjointHashIntSet with enough space for `N` sets initially.
  DisjointHashIntSet(const T N){
    rank.reserve(N);
    parent.reserve(N);
  }

  //Explicitly creates a set.
  void makeSet(const T n){
    if(rank.count(n)!=0)
      return;
    rank[n]   = 1;
    parent[n] = n;
  }

  bool isSet(const T n) const {
    return rank.count(n)!=0;
  }

  T findSet(const T n){
    makeSet(n);
    return _findSet(n);
  }

  T findSet(const T n) const {
    if(parent.count(n)==0)
      throw std::runtime_error("Could not get key from set!");
    return _findSet(n);
  }

  //Join two sets into a single set. Note that we "cannot" predict the `id` of
  //the resulting set ahead of time.
  void unionSet(const T a, const T b){
    const auto roota = findSet(a); //Find the ultimate parent of A
    const auto rootb = findSet(b); //Find the ultimate parent of B
    //Note that the foregoing collapses any chain of parents so that each set in
    //the chain points to the ultimate parent. Therefore, any subsequent call to
    //`findSet` involving any set in the chain will take `O(1)` time.

    //If A and B already share a parent, then they do not need merging.
    if(roota==rootb)         
      return;

    //If we always naively tacked A onto B then we could develop a worst-case
    //scenario in which each set pointed to just one other set in a long, linear
    //chain. If this happened then calls to `findSet()` would take `O(N)` time.
    //Instead, we keep track of how many child sets each set has and ensure that
    //the shorter tree of sets becomes part of the taller tree of sets. This
    //ensures that the tree does not grow taller unless the two trees were of
    //equal height in which case the resultant tree is taller by 1. In essence,
    //this bounds the depth of any query to being `log_2(N)`. However, due to
    //the use of path compression above, the query path is actually less than
    //this.

    if(rank[roota]<rank[rootb]){          //Is A shorter?
      parent[roota] = rootb;              //Make A a child of B.
    } else if(rank[roota]>rank[rootb]) {  //Is B shorter?
      parent[rootb] = roota;              //Make B a child of A.
    } else {                              //A and B are the same height
      parent[rootb] = roota;              //Arbitrarily make B a child of A
      rank[roota]++;                      //Increase A's height.
    }
  }

  //Using `unionSet` merges two sets in a way which does not allow us to decide
  //which set is the parent; however, `unionSet` helps guarantee fast queries.
  //`mergeAintoB` sacrifices speed but preserves parenthood by always making A a
  //child of B, regardless of the height of `B`.
  void mergeAintoB(const T a, const T b){
    makeSet(a);
    makeSet(b);
    parent[a] = b;
    if(rank[a]==rank[b]){
      rank[b]++;
    } else if(rank[a]>rank[b]){
      rank[b] = rank[a]+1;
    } else {
      //If `rank[b]>rank[a]` then making A a child of B does not increase B's
      //height.
    }
  }

  //Returns true if A and B belong to the same set.
  bool sameSet(const T a, const T b){
    return findSet(a)==findSet(b);
  }

  //Returns true if A and B belong to the same set.
  bool sameSet(const T a, const T b) const {
    return findSet(a)==findSet(b);
  }

  void clear(){
    rank.clear();
    parent.clear();
  }

  void shrink_to_fit(){
    *this = DisjointHashIntSet<T>();
  }

  const std::unordered_map<T,T>& getParentData() const {
    return parent;
  }
};



template<class T>
DisjointHashIntSet<T>& MergeDisjointHashIntSet(DisjointHashIntSet<T> &out, DisjointHashIntSet<T> &in){
  for(const auto &kv: in.parent)
    out.unionSet(kv.first, kv.second);

  return out;
}

#endif
