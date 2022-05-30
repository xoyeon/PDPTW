# 🚚 The pickup and delivery problem with time windows
<a href="https://confirmed-theater-e29.notion.site/Reading-List-48df9a2a6d614f648a64c4ca5e6054fe" target="_blank"><img src="https://img.shields.io/badge/Notion-ffffff?style=flat-square&logo=Notion&logoColor=black"/></a>


- [What are time windows?](https://desktop.arcgis.com/en/arcmap/latest/extensions/network-analyst/time-windows.htm)   
A time window is the period between a start and end time in which a network location, such as a stop in a route analysis, should be visited by a route.


## 🔗 목적
- Vehicle minimization
- Distance minimization
- minimize the number of routes

## 🔗 고려사항
- Dispatching
  - 1) Order matching : aims to find the best matching strategy between workers and demands
    - unserved demands
	- travel costs
	- worker availability (평균적으로 3건~5건)
  - 2) Fleet management : repositions idling workers to balance the local demand-supply ratio
    - heterogeneous fleet
	- flexible cargo size
- Routing
  - multiple depots
  - multiple time windows

## 🔗 주로 사용되는 알고리즘
- 휴리스틱  기법(Heuristic techniques)
- local search algorithm

## Papers to read


---

- [Google OR-Tools](https://developers.google.com/optimization/routing/pickup_delivery#python_1)
- [Li & Lim benchmark](https://www.sintef.no/projectweb/top/pdptw/li-lim-benchmark/)
- Instances for the Pickup and Delivery Problem with Time Windows based on open data
  - [article](https://www.sciencedirect.com/science/article/abs/pii/S0305054820301829?via%3Dihub)/[data](https://data.mendeley.com/datasets/wr2ct4r22f/2)
