# ACL2023_Control

This repository contains the supplementary files from: 

de-Dios-Flores, Iria, Juan Pablo García Amboage and Marcos García. 2023. Dependency resolution at the syntax-semantics interface: psycholinguistic and computational insights on control dependencies. *Proceedings of the  61st Annual Meeting of the Association for Computational Linguistics.*

It is composed of three folders: 

- 'datasets': contains the two datasets presented in the paper: acceptability (for experiments 1 and 2), and prediction (for experiment 3), with proper names (variants with pronouns can be found in each experiment folder inside 'code').

- 'code' includes the scripts and data needed to replicate experiments 2 and 3. Each subfolder includes a python script per language (compute_acceptability|prediction_spanish|galician.py). Running each script with the input file (e.g., "python compute_acceptability_spanish.py input_acceptability_names_spanish.csv") will generate a set of output files with the results. These output files are already available at the 'results' folder.

- 'results' has three subfolders, one per experiment. In 'experiment1' you can find the human acceptability means for each item in the dataset. In 'experiment2' and 'experiment3' you can find the output files (generated with the scripts in 'code') for each language (Spanish and Galician), condition (names and pronouns), and model (6 models for Spanish, 7 for Galician).
