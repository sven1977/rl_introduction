"""
 -------------------------------------------------------------------------
 20170820 - q_learning
 
 tabular Q-learning implementation on a simple example MDP
  
 created: 2017/08/10 in PyCharm
 (c) 2017 Sven Mika - ducandu GmbH
 -------------------------------------------------------------------------
"""

import random


class CrocodileLake(object):
    def __init__(self):
        self.s = None
        self.reset()

    def step(self, action):
        """
        performs a single step (action execution) in the environment

        :param str action: a valid action description (string)
        :return: a tuple with [new-state, reward, isTerminal]
        :rtype: Tuple[str,str,bool]
        """
        # our transition "function"
        s = None
        if action == "N":
            if self.s == "A":
                s = "B"
            elif self.s == "C":
                s = "D"
        elif action == "W":
            if self.s == "A":
                s = "C"
            elif self.s == "B":
                s = "D"
        elif action == "SWIM":
            if self.s == "A":
                s = "LAKE"
            elif self.s == "LAKE":
                s = "D"

        if s is None:
            raise Exception("ERROR: Undefined transition!")
        self.s = s

        # return s' (next state), reward, and terminal-indicator (se we can call reset)
        if s == "B":
            return s, -2.0, False
        if s == "C":
            return s, -2.0, False
        if s == "D":
            return s, 5.0, True
        if s == "LAKE":
            r = 0.0
            if random.random() > 0.9:
                r = -10.0
            return s, r, False

    def get_action_set(self, s=None):
        if s is None:
            s = self.s

        if s == "A":
            return ["N", "W", "SWIM"]
        elif s == "B":
            return ["W"]
        elif s == "C":
            return ["N"]
        elif s == "LAKE":
            return ["SWIM"]
        else:
            return []

    def reset(self):
        """
        resets this MDP
        """
        self.s = "A"


class QTable(object):
    def __init__(self, default=0.0):
        """
        :param List[str] action_space: a list of possible actions
        :param float default: a default value to use if an s/a-tuple does not exist in the table yet
        """
        self.default = default
        self.table = {}  # stores the Q-values for a given state and action tuple

    def get(self, sa):
        """
        returns the Q-value for any s/a-tuple (even it it does not exist yet in our table)

        :param sa: the s/a-tuple
        :return: the Q-value for the given s/a-tuple
        :rtype: float
        """
        if sa not in self.table:
            return self.upsert(sa)
        else:
            return self.table[sa]

    def upsert(self, sa, value=None):
        """
        updates OR inserts a Q-value for a given s,a-tuple

        :param Tuple[str,str] sa: the state/action tuple for which to update/insert a new value
        :param float value: the Q-value for the given s/a-tuple
        """
        if value is None:
            value = self.default

        self.table[sa] = value
        return value

    def get_best_a(self, s, action_set):
        """
        returns the best action (according to this Q-table) for a given state

        :param str s: the state to find the best action for
        :param List[str] action_set: the set of actions to search through (for the given state)
        :return: the best action for state s
        :rtype: str
        """
        a_star = None
        q_star = float("-inf")
        for a in action_set:
            t = (s, a)
            if self.get(t) > q_star:
                a_star = a
        return a_star

    def paint(self):
        """
        paints this table to the console
        """
        print("[s] /[a]  | [q-value]\n-----------------------")
        for (s, a), v in  self.table.items():
            print("{!s:>4}/{!s:4} | {}".format(s, a, v))


if __name__ == "__main__":
    # generate the MDP
    mdp = CrocodileLake()
    # get an empty Q-table with action set N, W, and swim
    q_table = QTable()

    # algo parameters:
    alpha = 0.1
    epsilon = 0.1

    # do 100 iterations of Q-learning
    for t in range(5000):
        s = mdp.s
        print("s=", s)
        a_set = mdp.get_action_set()
        # in epsilon % of th cases -> pick randomly
        if random.random() < epsilon:
            a = random.choice(a_set)
        # else -> pick best action according to our table so far
        else:
            a = q_table.get_best_a(s, a_set)
        print("a=", a)

        # do a single step and collect next state (s' == s_) and reward
        s_, r, is_terminal = mdp.step(a)
        print("s'=", s_, "r=", r)

        # do the q-update (for s and a, not for s' or a')
        key = (s, a)
        # old value
        Q = q_table.get(key)
        print("old Q=", Q)
        # new value
        Q_new = (1.0 - alpha) * Q + alpha * (r + q_table.get((s_, q_table.get_best_a(s_, mdp.get_action_set(s_)))))
        print("new Q=", Q_new)
        q_table.upsert(key, Q_new)

        # if we reached the end -> reset our MDP and start again
        if is_terminal:
            mdp.reset()

        # slowly reduce epsilon and alpha (to fulfil the guaranteed-convergence condition)
        # - if we don't do this, this algorithm will take many more iterations to converge to the optimal policy/q-table
        # - adding these two lines and setting the initial values to somewhat close to 1.0 will reduce the number of iterations to ~100
        #epsilon *= 0.995
        #alpha *= 0.995

        print("e=", epsilon, "alpha=", alpha)
        print()

    # show the final Q-table
    q_table.paint()