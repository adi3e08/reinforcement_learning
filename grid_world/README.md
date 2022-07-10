# Grid World

Consider the classic problem of finding the shortest path between two locations on a grid filled with obstacles.

<p align="center">
<img src="https://adi3e08.github.io/files/blog/grid-world/imgs/grid.png" width="100%" height="100%"/>
</p>

**Given**

- Grid Dimensions : Height $H$, Width $W$

- Start location : $(h_{S},w_{S})$

- Goal location : $(h_{G},w_{G})$

- Obstacles :  $\mathcal{X} = $ { $(h_{X}^{i},w_{X}^{i}):  1 \leq i \leq N_{X}$ }

- Allowed moves : King's moves

We consider two variations of value iteration,
<p align="center">
<img src="https://adi3e08.github.io/files/blog/grid-world/imgs/value_iteration_sync.png" width="90%" height="90%"/>
</p>

<p align="center">
<img src="https://adi3e08.github.io/files/blog/grid-world/imgs/value_iteration_async.png" width="90%" height="90%"/>
</p>

## Result
<p align="center">
<img src="https://adi3e08.github.io/files/blog/grid-world/imgs/result.png" width="75%" height="75%"/>
</p>

<p align="center">
<img src="https://adi3e08.github.io/files/blog/grid-world/imgs/best_path.png" width="100%" height="100%"/>
</p>

You can find the code used in these experiments [here](https://github.com/adi3e08/tabular_RL/tree/main/grid_world).
