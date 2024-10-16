# ECG Disorder Classification with Mixture of Experts

## Overview

This repository contains the implementation of the paper **"[Leveraging Cardiologists' Prior Knowledge and a Mixture of Experts Model for Hierarchically Predicting ECG Disorders](<https://cinc.org/2024/Program/accepted/477_Preprint.pdf>)."** The study proposes a Mixture of Experts (MoE) model to identify six physician-defined clinical labels related to rhythm and conduction disorders in Electrocardiograms (ECGs). 

## Main Idea

The MoE framework combines the opinions of multiple models (experts) to make more accurate predictions by leveraging prior knowledge from cardiology specialists. This approach organizes disorders into a two-level hierarchy, enhancing the classification process by tailoring the MoE architecture to incorporate this medical knowledge.

## Repository Contents

- **Code Implementation**: The source code for training the MoE model on the CODE-15 dataset.
- **Results**: Evaluation metrics demonstrating the model's performance compared to standard classification approaches.

## Acknowledgments

This work was partially funded by CNPq, CAPES, and FAPEMIG, as well as CIIA-Saude. We also thank the Telehealth Center of Minas Gerais for access to the data and the productive discussions within the scope of this work
