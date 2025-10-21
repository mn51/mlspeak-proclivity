# Learning Time-Varying Turn-Taking Behavior in Group Conversations
by Madeline Navarro, Lisa O'Bryan, and Santiago Segarra.

This code belongs to a paper that has been submitted to *ICASSP 2026*.

## Summary

We propose a flexible probabilistic model for *predicting turn-taking patterns in group conversations* based solely on *individual characteristics* and *past speaking behavior*.
Many models of conversation dynamics cannot yield insights that generalize beyond a single group.
Moreover, past works often aim to characterize speaking behavior through a universal formulation that may not be suitable for all groups.
We thus develop a generalization of prior conversation models that predicts speaking turns among individuals in any group based on their individual characteristics, that is, personality traits, and prior speaking behavior.
Importantly, our approach provides the novel ability to *learn* how speaking inclination varies based on when individuals last spoke.
We apply our model to synthetic and real-world conversation data to verify the proposed approach and characterize real group interactions.
Our results demonstrate that previous behavioral models may not always be realistic, motivating our data-driven yet theoretically grounded approach.

## Software implementation
All source code comprising our model are located in the `src` folder.
Experimental results were obtained using the experiments in the `notebooks` folder.
