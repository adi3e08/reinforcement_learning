# Grid World

Consider the classic problem of finding the shortest path between two locations on a grid filled with obstacles.

![Grid](https://adi3e08.github.io/files/blog/grid-world/imgs/grid.png)

**Given**

1.  Grid Dimensions : Height H, Width W

2.  Start location : (h<sub>S</sub>, w<sub>S</sub>)

3.  Goal location : (h<sub>G</sub>, w<sub>G</sub>)

4.  Number of Obstacles : N<sub>X</sub>

5.  Obstacle locations :
    X = { (h<sub>X</sub><sup>i</sup>, w<sub>X</sub><sup>i</sup>) :  0 <= i <= N<sub>X</sub>-1 }

6.  Allowed moves : King's moves

We consider two variations of value iteration

![value_iteration_sync](https://adi3e08.github.io/files/blog/grid-world/imgs/value_iteration_sync.png)

![value_iteration_async](https://adi3e08.github.io/files/blog/grid-world/imgs/value_iteration_async.png)

**Result**

![result](https://adi3e08.github.io/files/blog/grid-world/imgs/result.png)

![best_path](https://adi3e08.github.io/files/blog/grid-world/imgs/best_path.png)
