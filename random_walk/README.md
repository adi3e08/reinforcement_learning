# Random Walk

Consider N states laid out in a row, with the left most state being a terminal state. Consider an agent who, when at a non-terminal state, chooses to move with probability p to the left neighbour and with probability 1-p to the right neighbour(or to itself in case of the right-most state). All rewards are 1.

![MDP](https://adi3e08.github.io/files/blog/random-walk/imgs/mdp.png)

![action_space](https://adi3e08.github.io/files/blog/random-walk/imgs/action_space.png)
  
In the solutions to follow, we store the Value function *V* in a table
since the State and Action Spaces are discrete and finite.

**Model Based Policy Evaluation**

![policy_eval_sync](https://adi3e08.github.io/files/blog/random-walk/imgs/policy_eval_sync.png)

![policy_eval_async](https://adi3e08.github.io/files/blog/random-walk/imgs/policy_eval_async.png)

![model_based_policy_eval](https://adi3e08.github.io/files/blog/random-walk/imgs/model_based_policy_eval.png)

**Model Free Policy Evaluation**

![td_lambda](https://adi3e08.github.io/files/blog/random-walk/imgs/td_lambda.png)

![offline_lambda](https://adi3e08.github.io/files/blog/random-walk/imgs/offline_lambda.png)

![model_free_policy_eval](https://adi3e08.github.io/files/blog/random-walk/imgs/model_free_policy_eval.png)