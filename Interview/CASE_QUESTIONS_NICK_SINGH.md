# The End-to-End ML Workflow as given in Nick Singh's book - Ace the Data Science Interview


## STEP 1: CLARIFY THE PROBLEM AND CONSTRAINTS

### Clarification
- What is the dependent variable? What is/are the input variables
- How has the problem been approached in the past by the business. Is there a baseline that we have to beat. By how much
- Is ML even needed? Would simple heuristics or rule-based approaches work? 
- Is it legal or ethical to apply ML to this problem
- How would the end user benefit from the solution? How would they use the solution (integration with existing solution or standalone)
- Is there a clear value add to the business?
- How does incorrect predictions impact the business? (eg. it is fine to get a spam email rather than a high-risk application being auto-approved)


### Technical requirements
- What is the latency needed?
- Throughput requirements?
- Where is the model being deployed? How costly is the deployment?

## STEP 2: ESTABLISH METRICS
- Pick simple, observable metrics
- What are the KPIs (eg.  time to resolve a request)
- It's best to opt for a single metric rather than picking multiple metrics to capture different sub goals. That's because a single metric makes it easier to track model performance. Plus, it's easier to figure out how to optimize a single number. 
- It’s better to start your explanation with a single metric, but then hedge your answer by mentioning other potential metrics to track.

- Suggest balancing things instead of one, by weighing different sub-metrics, such as false positives versus false negatives to create a final metric to track. This is often known as an **OEC (overall evaluation criterion)** and gives you a balance between different metrics.

- **Establish what success looks like**. While for a classifier you might desire 100% accuracy, this is a **realistic bar** for measuring success? Is there a **threshold** to be good enough? 
- If possible, you should use the performance of the **existing setup for comparison** for example, if the average time to resolution for customer support tickets is five hours, you could aim for four hours at a 90% ticket classification accuracy. 


- Be sure to voice out these metric considerations to your interviewer so that you can show you’ve thought critically about the problem. 



## STEP 3: UNDERSTAND DATA SOURCES

## Start
- "garbage in, garbage out." 
- While it’s intuitive to use the internal company data relevant to the problem at hand, that’s not the be-all end-all of data sourcing.

### Think outside the box regarding what data sources to consider

- Can you acquire more data by **crowdsourcing** (e.g., Amazon Mechanical Turk)?
- Can you ask users for data as part of the **user onboarding process**?
- Can you **buy second- and third-party datasets**?
- Can you **ethically scrape** the data from online sources?
- Can you send your unlabeled internal data off to a labeling and **annotation service**?


### Edge Cases
- Intentionally source **more examples of edge cases you find representative, so called active learning**. For example, suppose your traffic light detection system has poor performance on **snowy days**. You could add a session of your training images that has examples in snowy conditions. Simulations are used in the **autonomous vehicle** field because encountering the volume of **rare and risky scenarios** needed to adequately train a model based on only real-world driving is infeasible.

### Questions to consider

- How fresh is the data? 
- How often will the data be updated?
- Is there a data dictionary available? Have you talked to subject matter experts about it?
- How was the data collected? Was there any sampling, selection, or response bias?


## STEP 4: EXPLORE THE DATA
### Profiling
- profile the columns at a high level. 
- What might be useful? Which ones have practically no variance and thus may not be of predictive value? 
- Which columns look messy? 
- Which have a lot of missing or null values? 
- Also look at summary statistics like the mean, median, and quantiles.

### Viz
- Certain features (e.g., age, weight, price) can be visualized with a histogram through binning. 
- Visualize the range of continuous and categorical variables. 
- Analyze the relationships between variables using a correlation matrix. This can help you quickly spot what variables are correlated with one another, as well as what might be correlated with the target variable at hand.


## STEP 5: CLEAN THE DATA
- Issues include missing values, data entry errors, date merging issues, different data formatting, column changes
- dropping irrelevant data or erroneous, duplicated rows entirely
- handle incorrect values that don’t match up with the supposed data schema. 
- For example, for fields containing human-entered data, there is often a pattern of error in what you could find and then use to clean up.


### Missing Values
- **Imputing** the missing values via basic methods such as column mean/median
- Using a **model or distribution to impute the missing data**
- **Dropping the rows** with missing values (as a last resort)


### Outliers
- like manual data entry issues or logging hiccups. 
- Or maybe they accurately reflect the actual data. 
- Outliers can be removed outright, truncated, or left as is, 

-  A 4-year-old human wouldn’t be strange, a 5-foot-tall person isn’t odd, but a 4-year-old that’s 5-feet tall would either be an outlier, data entry error, or a Guinness world record holder.



## STEP 6: FEATURE ENGINEERING
### Usual techniques
- Art than science
- Feature Selection on domain knowledge
- Feature transformation - capping, flooring, log, powers
- Binning
- Dimensionality Reduction
- Standardising - minmax scaling / standard scaling
- Categorical data: 
   - One-hot encoding
   - Categorical encoding
   - Hashing
   - Embedding


### NLP
- Stemming, Lemmatization, Filtering stop words
- Bag of words - represents text as a collection of words associating each word with its frequency
- N-grams - extension of BoW where we use N words in a sequence
- Pretrained Embeddings- word2vec, GloVE




## STEP 7: MODEL SELECTION
- Training and prediction speed- linear regression can be quicker than neural networks
- Budget - Neural networks are computationally and cost intensive
- Volume of data - NNs can handle large amount of data and more dimensions also
- Categorial vs Numerical - Text features would need encoding
- Explainability

## STEP 8: MODEL TRAINING AND EVALUATION
- Train test validation split
- Cross Validation
- Hyperparameter tuning
- Feature importances? Data Imbalances
- If training too slow - sampling techniques. 
    - Random sampling
    - Stratified sampling
    - Undersampling oversampling for imbalanced
    - SMOTE


- Need Pyspark? 
- GCP provides SQL-like training
- 

## STEP 9: DEPLOYMENT
- MLOps
- Airflow MLFlow DVC Git 
- Wandb


- Online deployment - less latency, real-time predictions, caching layer, computationally intensive
- Batch predictions - Don't need immediate results (like recommendation systems)
- FastAPI Docker Prometheus Grafana
- Logging to monitor

### Issues
- Model degradation
- Data drift
- Training-serving skew: time and performance hit between model in training and serving time
- How much to retrain your model
- Events that might trigger model refresh
- How much to rely on historical data




## STEP 10: ITERATE
- Look at bad predictions - bucket them by reasons