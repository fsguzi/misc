# schedule2020 rough timeline

## 5.20-5.22
* understanding problem
    * easy part: resource cap, max apps assignment on same node
    * confusing part: socket/core/cpu

## 5.23
* thinking about approaches
    * static problem: use some form of MIP/LP plus some heuristic
    * reschedule problem: maybe interpret as shortest paths for some possible layout, most likely pure heuristic
* thinking about implementation
    * using GO
    * static part need simplex/b&b implement
    * heuristics should not be hard

## 5.24
* implement straightforward MIP/LP formulation for static problem w/o core interaction part(with SCIP in py)
    * variable for each <app,node> pair
    * integer solution likely not practical to find
    * relaxation solves reasonably in about 10 minutes(no optimal)

## 5.25
* reading stuff on implementing simplex
    * tricks in revised simplex: product form of basic columns inverse
* experiment a column generation formulation for static problem
    * variable for each {[apps],node_type} pattern
    * create initial patterns with heuristic -> LP relaxation -> add columns until LP solved
    * with initial patterns: master problem relax solves instantly, integer optimal found in few seconds

## 5.26
* manually add column generating subproblem
    * solved as individual exact MIP
    * new columns fed to master problem which reconstructs and restarts

## 5.27
* try reusing same problem instance for subproblem(only different obj coefficient)
    * clear solution and preprocess data then change objective
    * success
* try reusing same problem instance for masterproblem
    * issue with dual solution fetch returns invalid results
    * fail
* try SCIP's embedded method for CG pricer
    * final results different from manual column generation
    * more columns and better optimal value
    * master LP solves in ~10mins

## 5.28
* try to find ways to speed up CG fomulation as whole
    * discover subproblem solve slower towards the end
        * define different obj than vanilla most reduced cost?
        * solve with custom made algo?
    * degeneracy issue, unchanged obj in many rounds
        * perturbation? different pivot rule?
* try using c++ interface of SCIP which has complete functionality
