# Thompson-Sampling-for-a-Fatigue-aware-Online-Recommendation-System
## Introduction

#### This python code is for paper ''Thompson Sampling for a Fatigue-aware Online Recommendation System''. Please find details on [https://arxiv.org/abs/1901.07734](https://arxiv.org/abs/1901.07734)

#### This code implements algorithm 1, 2, 3. It is used for performing the experiments in Section 7.

#### The result file and all the charts are the result when we run the code.

* Based\_define\_function.py: This code provide the basic function used in the paper, i.e. calculate the order of messages, the users feedback, the total payoff, etc. 

* Thompsonbasedpy.py: This code provides function thompson\_base that we implement algorithm 1.

* TSgaussian.py: This code provides function TSgaussian\_base that we implement algorithm 1.

* UCBV.py: This code provides function ucbv that we implement algorithm 3.

* ucbbasedpy.py: This code provides function ucb\_base that we implement algorithm in [Cao and Sun (2019)](https://arxiv.org/abs/1903.08193).

* experiment.py: This code calls the function from other py file. The current experiment.py file is that we call algorithm 2 to do 10 times experiments using different R alpha and beta. We record the result in txt file and pickle down in result file.

* testfile.py: This code plots the chart of the result of each algorithm in the result file. The current testfile.py file is that we plot the result chart of algorithm 1 when u is generated from [0, 0.5].

