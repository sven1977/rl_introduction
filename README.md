# Simple Q-Learning Implementation
q_learning.py is the script accompanying the talk
"Reinforcement Learning, An Introduction", Dr. Sven Mika
Duesseldorf Germany Aug 20th 2017

#### 1) Run the code:
`python q_learning.py` (python3 only)


#### 2) Alter the parameters of the algo:

Play around with the epsilon and alpha parameters to lower the number of
necessary iterations to a minimum. Also, if you start with large values
(close to 1.0) for both and slowly reduce both values of time (by
multiplying with a factor < 1.0 each iteration), you will get better
results (faster convergence). In the end, your table should give
you the expected accumulated future rewards for each state/action pair.


#### 3) Expected Output:

For epsilon=alpha=0.1 (no reduction of these over time)
and 5000 iterations, you should get something like:

```
[s] /[a]  | [q-value]
-----------------------
   A/N    | 3.0
   B/W    | 5.0
   D/None | 0.0
   A/W    | 3.0
   A/SWIM | 3.4
LAKE/SWIM | 5.0
   C/N    | 5.0
```

