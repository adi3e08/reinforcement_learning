Consider the classic problem of finding the shortest path between two cells on a grid filled with obstacles.

![Grid](https://adi3e08.github.io/files/blog/grid-world/imgs/grid.png)

**Given**

1.  Dimensions : Height $H$, Width $W$

2.  Start location : $(h_{S},w_{S})$

3.  Goal location : $(h_{G},w_{G})$

4.  Number of Obstacles : $N_{X}$

5.  Obstacles :
    $\mathcal{X} = $ { $(h_{X}^{i},w_{X}^{i}):  0 \leq i \leq N_{obs}-1$ }

6.  Allowed moves : King's moves

We consider two variations of value iteration

![value_iteration_sync](https://adi3e08.github.io/files/blog/grid-world/imgs/value_iteration_sync.png)

![value_iteration_async](https://adi3e08.github.io/files/blog/grid-world/imgs/value_iteration_async.png)

**Result**

![result](https://adi3e08.github.io/files/blog/grid-world/imgs/result.png)

![best_path](https://adi3e08.github.io/files/blog/grid-world/imgs/best_path.png)
