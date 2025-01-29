# Deep Funding Machine Prediction and Cartography with the Omniacs.DAO

Our submission to the Deep Funding Mini-Contest leverages a high level feature embedding of the funded repositories in combination with standard best practices for modelling tabular data presented with a novel “Funding Map” for added insight into the funding performance of the projects on interest.

* **Part 1**: Submission under “Omniacs.DAO” on Hugging Face with an MSE of 0.0145
* **Part 2**: Submission under “Omniacs.DAO” on CryptoPond with an MSE of 0.0564
* **Code and Data**: https://github.com/OmniacsDAO/deepfundSubmission

**Executive Summary**

* We used a combination of Github repo features in addition to an Nomic embedding representation of the repo readme files as inputs into a grid search optimized gradient boosting machine to achieve a top 10 placement on both part 1 and part 2 of the mini-contest.
* We then applied linear and non-linear dimensionality reduction techniques to the embeddings to generate “Funding Maps” that highlight not only groups of similar repositories but also their funding amounts, highlighting both under and over performing projects.
* As a contribution to the Deep Funding community, we open sourced all of our post processed data for easier replication of our modeling efforts. Our code has also been published as well.

**Approach**

In a quick "cookbook" format, our approach to developing models first consisted of collecting both training datasets from the HuggingFace and CryptoPond platforms. We then supplemented the data with information from the Open Source Observer using Bigquery, as well as, both procuring the project readme.md files via the github API and then extracting a vectorization of the text using the ```nomic-embed-text:v1.5``` embedding model.  We then utilized a grid search to find the optimal hyperparameters for a standard gradient boosted model. This approach resulted in top 10 placements on both leaderboards at the time of writing.

**Deep Funding Cartography**

To take our results one step further we leveraged manifold learning as a dimension reduction technique to create what we are calling a "Funding Map" of the projects included in the training set, but mapped out by the similarity of their readme.md files. These readme files, when vectorized, can serve as quantitative proxies for similarity once combined with a distance metric like  Euclidean distance.  Before the creation of the map; however, we had to first derive a proxy ranking of the funding performance of the repos.  This was done by using a simple sum of the predicted weights from our model.

![image](https://github.com/user-attachments/assets/4193b8bc-ce98-46ac-b19b-a628d9a644f7)

This allowed us to apply a simple coloring scheme where the top 20% are labeled green while the bottom 20% are labeled red.  The resulting "Funding Map" is below.

![image](https://github.com/user-attachments/assets/6aa89d30-7da3-4b78-8a8e-791814df159b)

One really cool insight highlighted from the map is that related repos don't necessarily receive the same amount of funding.  

![image](https://github.com/user-attachments/assets/0d7a137a-787e-415d-9f82-280f66576922)

In the section of the map above, the Coinbase Wallet SDK is highlighted as a green projects, while the WalletConnect monorepo designed to "Open protocol for connecting Wallets to Dapps" severely underperforms. We can see this not only from the raw scores...

![image](https://github.com/user-attachments/assets/f260d6d4-0abb-4dae-9433-af9fc71d02e8)

... but also the head to head match-ups in the training data.

![image](https://github.com/user-attachments/assets/be767314-9028-4d76-9b87-7f387ed26e26)

![image](https://github.com/user-attachments/assets/9dff23f0-99cd-4e12-8dd7-c39250fdc8ce)

We took some extra time to bring our analysis to life by creating an [interactive version of our Funding Map](https://apps.omniacsdao.xyz/deepfunding/) so others can explore the donation space we derived.

![cartographyOD](https://github.com/user-attachments/assets/121eb902-d735-4362-856c-d2517c820847)

In conclusion, we think our modeling efforts and statistical visualizations gets us all one step closer to more deeply understanding public goods funding and how we can better allocate capital to what matters in the space!
