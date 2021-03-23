EE 660  Final Project Report  12/03/2020 

Compare Semi-Supervised Learning and Supervised Learning in Rain Prediction in Australia  

EE 660 Course Project 

**Project Type:**  Design a system based on real-world data Po Yu Yang     [ poyuyang@usc.edu ](mailto:poyuyang@usc.edu) 

Liang Chun Chen [ lianchu@usc.edu ](mailto:lianchu@usc.edu)

Date: 12/ 03/ 2020 

1. Abstract 

In this project we will give a model to tell people in Australia who need to bring their umbrella or not tomorrow. To be concise, it is a binary classification which predicts whether it will rain tomorrow in Australia. We are going to use Semi-Supervised Learning  to  compare  with  our  Supervised  Learning.  Moreover,  we  are  going  to compare with others' existing results. In Semi-Supervised Learning, we are going to apply  the  Label  Propagation  and  Label  Spreading  approach.  Compared  with supervised learning, we are going to implement CART, Random Forest and Adaboost models. In the end, we give the best model and the comparison between SSL and SL in Rain in Australia dataset. 

2. Introduction 
1. Problem Type, Statement and Goals 

In this problem, we are going to tell people whether they need to bring an umbrella or not in Australia. The goal is to predict whether it will rain or not tomorrow  according  to  our  data.  Moreover,  we  will  implement  Semi- Supervised Learning algorithms to compare with the Supervised learning model.  Our  goal  is  to  let  Semi-Supervised  Learning  work  as  well  as Supervised Learning algorithms. This is a classification problem with two classes (rain tomorrow or not). The data used in this project was published on Kaggle which was collected by Joe Young. [1] According to the dataset, it has 142193 data, 23 features, and one label. In this classification problem, we need to deal with the amount of missing data and unbalanced dataset.  

2. Literature Review (Optional) 

Some existing approaches work on Supervised Learning, and most focus on Logistic Regression models. We will compare our model and our previous work. In this project, we will focus on semi-supervised learning compared with supervised learning. 

3. Our Prior and Related Work - None 
3. Overview of Our Approach 

To begin with, we handle the missing data issue and imbalance problem and encode  the  category  features.  Then,  we  split  the  data  into  training, validation, and Testing. We did the standardization on training data. After the whole preprocessing process and feature selection, we implemented different machine learning models on training data and used a validation dataset to find out the best hypothesis mode. The machine learning models included Supervised learning and Semi-Supervised learning. We will plot the ROC curve and confusion matrix on testing data to analyze the result. In the end, we applied our model on testing data to analyze the performance and get the best model on the rain prediction in Australia.  

3. Implementation 

In this section, we will briefly describe our data set and explain all the details in each approach in the project.  

1. Data Set 

This dataset contains about 10 years of daily weather observations from numerous Australian weather stations since 2008. There are 23 features and 1 label in this dataset. The data information is shown below: 



|**#** |**Features** |**dtype** |**#** |**Features** |**dtype** |
| - | - | - | - | - | - |
|0 |Date |String |12 |WindSpeed3pm |Integer |
|1 |Location |Category |13 |Humidity9am |Integer |
|2 |Min Temp |Float |14 |Humidity9am |Integer |
|3 |Max Temp |Float |15 |Pressure9am |Float |
|4 |Rainfall |Float |16 |Pressure3pm |Float |
|5 |Evaporation |Float |17 |Cloud9am |Integer |
|6 |Sunshine |Float |18 |Cloud3pm |Integer |
|7 |WindGustDir |Category |19 |Temp9am |Float |
|8 |WindGustSpeed |Integer |20 |Temp3pm |Float |
|9 |WindDir9am |Category |21 |RainToday |Binary |
|10 |WindDir3pm |Category |22 |RISK\_MM |Float |
|11 |WindSpeed9am |Integer ||||
The unit of the temperature in this dataset used Celsius (Feature 2, 3, 19, 20).   The  wind  direction  features  are  contained  in  the  16  directions  (Feature 7, 9, 10).  

2. Dataset Methodology 

We  split  the  dataset  into  training,  validation,  and  testing  parts.  The percentage of how many data on each part shown below: 



||**Training** |**Validation**  |**Testing** |
| :- | - | - | - |
|ratio(amount) |64%(141204) |16%(35301) |20%(44127) |
Before splitting the dataset, we increase the dataset so that the data is balanced.  Then  we  deal  with  the  missing  information  and  encode  the category feature. After the below preprocessing, we split the dataset into training,  validation,  and  testing  parts.  In  this  project,  we  apply  normal validation  instead  of  using  cross-validation.  The  reason  is  that  we  have enough data to get a better result. Moreover, cross-validation costs more time than standard validation. 

3. Preprocessing, Feature Extraction, Dimensionality Adjustment Preprocessing**:** 

In the preprocessing step, we handle the missing data information and the imbalance problem. Also, we need to turn the category feature into numbers. We did these three steps before we split the dataset into training and testing.  

In the beginning, we handle the class imbalance. The raw data have just 22% data that predict tomorrow will rain, and 78% data show that it will rain tomorrow. Thus, we can get less information from the label of raining tomorrow. In this situation, we decided to increase the dataset by randomly duplicating the data from the original dataset which is raining tomorrow. Thus, the label can be balanced and get better performance. Here is the image before and after we handle the class imbalance problem. 

![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.001.png)![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.002.png)

Original dataset  After handling imbalance problem 110910 : 31283  110316 : 110316 

Secondly, we encode the category features (Location, WindGustDir, WindDir9am, WindDir3pm, RainToday, RainTomorrow). In the dataset of rain in Australia, we get 49 cities in Australia including Canberra, Syden, etc. Thus, we label the whole city by 0 to 48. In the feature that relates to wind direction, we label the direction by 0 to 15 (wind comes from 16 directions). Lately, we encode the binary features (rainTodday, rainTomorrow). 

Thirdly, we handle the missing data in two parts. In the first part, we deal with the category type of feature. We fill the missing data with the highest frequency word in that feature. In the second part (numerical info), we determine the missing data by the other feature. We apply IterativeImputer that estimates each feature from neighbors or close features in a round-robin fashion(iterate). [3] 

Lastly, we do the standardization on training data (after splitting the dataset). To avoid data snooping, we just do the standardization on the training dataset and transform validation data and testing data by training model. In the category feature, it is unnecessary to do the standardization. 

Feature Selection: 

According to the article, we know that the feature ‘Risk-MM’ is the amount of rainfall in millimeters for the next day. Thus, we need to drop this feature. Also, we ignore the ‘data’ feature which is not useful for rain prediction. After dropping these two features we observe each feature correlates with another feature. If the two features have a high correlation, we can tell that these two features have a similar distribution. The correlation table below shows: 

![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.003.png)

Correlation Heatmap 

As we can see ‘MinTemp’ and ‘Temp9am’, ‘MaxTemp’ and ‘Temp3pm’, ‘MinTemp’ and ‘MaxTemp’, ‘Pressure3pm’ and ‘Pressure9am’, ‘WindGustSpeed’ and ‘WindSpeed3pm’ have higher correlation. Thus, we can use this information to reduce the features. The final features we chosen will be: ‘Location’, ‘MinTemp’, ‘Rainfall’,   ‘Evaporation’,  ‘SunShine’, ‘WindGustDir’, ‘WindDir9am’, ‘WindDir3pm’, ‘WindSpeed9pm’,  ‘Wind Speed3pm’, ‘Humidity9am’, ‘Humidity3pm’, ‘Pressure3pm’, ‘Cloud9am’, ‘Cloud3pm’, ‘RainToday’ 

4. Training Process 

In the Training Process, we separate the training data into training and validation parts. In this section, we will implement four supervised learning and two semi-supervised learning methods. In supervised 

learning, we implement logistic regression, decision tree, Random forest, and Adaboost (These four models are all included in the EE660 course). In semi-supervised learning, we are going to modify the input to get the unlabeled data, and round the LabelPropagation and LabelSpreading algorithms. To find the best parameter on each model, we will use validation data to find the best model. 

Supervised Learning (SL): 

In Supervised learning, we implemented three models which are Decision tree, Random Forest, and Ababoost. We also try the best model of logistic regression by other people who work on Kaggle. [2] 

- *Decision Tree (CART):* 

In the Decision tree, we choose one feature to divide the dataset into two regions in each iteration. when the dataset has all been split into two classes or the max depth of the tree is larger than the constraint. To implement the Decision Tree algorithm, we apply the sklearn DecisionTreeClassifier model to complete the whole process. We harness the Gini criterion to decide which split next, and we find the best parameter(max\_depth) by the result of validation data. We tried the max depth from 2 to 50 and found the best performance shown on validation data. The disadvantage of the decision tree is that because CART is a Greedy approach, we might not get the best result. The result is easily overfitting. Thus, we implement the Random Forest.  

- *Random Forest:* 

The Random Forest approach is an extension of CART, we draw n dataset D’ on training data D, and create each tree of each D’. Finally, we get the result by averaging the result from all of the trees. We implemented a sklearn random forest model, and tried different numbers of trees. The parameter is also found by the result of validation. For the max depth of each tree, we use the best max depth we found in the Decision tree. We tried to build 30 to 60 trees and see the performance on the validation set. Compared with Decision Tree and Random Forest, Random forest lowers the variance and lowers the risk of overfitting. 

- *Adaboost:* 

Adaboost is an adaptive basis function model. We create stumps to classify the data. Each stump we influence to the next stump, thus, we will keep updating the weight of each stump. In the end, we will get the best model. We implement Adaboost by sklearn model, and use decision tree estimate set up max depth equal 2 to run the whole process. However, we did not get the better result on Adaboost in validation data. Thus, we tried to increase the max\_depth of Adaboost, and the performance worked pretty well.  

Semi Supervised Learning (SSL): 

Unlike supervised learning, we cannot handle that much data to semi supervised learning, therefore, we shrink the training data to 50000 then split the training data into two parts, and change one part of the label into -1 so that the models know they are unlabeled data. 

We decide to use ‘knn’ for the kernel because we have a large data set. ‘knn’ is faster and gives sparse representation. ‘rbf’ is harder to compute and store a dense transition matrix in memory, 

- *Label Propagation:* 

The algorithm will create a fully connected graph where nodes are all the data points. [4] The edges between nodes will be weights: 

2

- exp⁡( , 

)

2

where d - distance function (Euclidean in this case, but in general could be any distance function you want),  - hyperparameter, that controls weights. 

.

, =⁡∑ +

=1 ,

1. T is just a matrix with probability of every data point to be in class C 
1. Propagate Y <- TY (we "propagate" labels from labeled data to unlabeled) 
1. Row-normalize Y (value of element in row / sum of all elements values in row) 
1. Clamp the labeled data (we fix our labeled data, so our algorithm won't change probability of class, or in other words change the label) 
1. Repeat from step 1 until Y converges (we recalculate our distances and weights, that will give us a different transition matrix that will change our belief in assigned labels, repeat until the process converges). 

We implemented sklearn Label Propagation to complete the whole model training. 

- *LabelSpreading:* 

It will create an affinity matrix 

−|| − ||2

- exp⁡( )

2

Then we will construct the matrix (Laplacian):⁡ =⁡ −21 −21

where D - diagonal matrix with its (i, i)-element equal to the sum of the i- th row of W. 

Iterate  ( + 1)⁡= ⁡ ( ⁡)⁡+ (1 − ) ⁡where α is a parameter in (0, 1), F(t) - classifying function. 

During each iteration of the third step each point receives the information from its neighbors (first term), and also retains its initial information (second term). The parameter α specifies the relative amount of the information from its neighbors and its initial label information. Finally, the label of each unlabeled point is set to be the class of which it has received most information during the iteration process. [4] 

We implemented sklearn Label Spreading to complete the whole model training. 

5. Model Selection and Comparison of Results 

Following our Train process, we can use a validation dataset to find out the best parameters for each model. Here we showed our result of each model and their best parameters. We also apply our best model of each machine learning  approach  into  the  testing  dataset,  and  plot the  ROC  curve  and confusion  matrix  to  analyze  the  performance  of  each  mode  in  the  next section. 

Decision Tree (Model selection/ training vs validation) ![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.004.png)

|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.005.png)|
| - |
|The best parameter (Max\_depth): 45 |
|Training accuracy: 99.98% |Validation acc:  88.7% |
|We tried the max\_depth between 2 to 50 to get the best depth for the tree, after max\_depth goes beyond, the accuracy of the validation data seems to converge at 88.7%. Thus, the best max\_depth will be 45. It might be overfitting when we keep increasing the max\_depth.  |


|Random Forest (Model selection/ training vs validation) |
| - |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.006.png)|
|The best parameter (n trees): 70 |
|Training accuracy:  99.98% |Validation acc: 92.92% |
|We tried to use the max\_depth we get from the decision tree and the accuracy of validation data converged at 92.92%. Thus, the best parameter that we get is 70 trees for random forest. It will overfit if we do more trees. |
|AdaBoost (Model selection/ training vs validation) |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.007.png)|
|The best parameter(number of stumps): greater than 14000 |
|Training accuracy: 95.42% |Validation accuracy: 87.32% |
|To get the best model on Adaboost, we increase the number of stumps to get better accuracy. However, the performance increases slowly and the run time increases extremely when the stumps increase. We try 14000 stumps to fit the model and get 87% accuracy on validation data. It cost too much to do more stump. Thus, we stop the iteration in 14000 stumps, and do the modification below. |


|AdaBoost (Model selection/ training vs validation) (modification) |
| - |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.008.png)|
|The best parameter: depth = 23, trees = 50 |
|Training accuracy: 99.98% |Validation accuracy: 93.11% |
We chose the tree parameter from random forest results and tried to get the best max\_depth from ![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.009.png)2 to 25. It comes out that the validation accuracy starts to converge after max\_depth equal 20. The best parameter we got is 23, we are afraid that the result will be overfitting if we keep adding it up. 



|LabelPropagation (Model selection/ training vs validation) |
| - |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.010.png)|
|The best parameter(n\_neighbors): 20 |
|Validation accuracy: 77.28% |
|We tried the n\_neighbors from 10 to 100, then we found out that after 40, the accuracy of validation data starts to go down, so we shrink the range to 10 to 40 to find a better result at n\_neighbors = 20.  |
Label Spreading (Model selection/ training vs validation) 

![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.011.png)

|The best parameter(Alpha): 0.03 |
| - |
|Validation acc: 77.55% |
|We used the same n\_neighbors we found at LabelPropagation and tried to find alpha from 0.01 to 0.1, because after 0.1 the result of validation accuracy also starts to go down. |
4. Final Results and Interpretation 

In this section, we are going to separate into two parts. First, we compare different model’s performance on supervised learning. Then, we will compare the supervised learning model and semi-supervised learning.  

**Supervised Learning:** 



|**Logistic Regression** |**Decision Tree**  |**Random Forest** |
| - | - | - |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.012.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.013.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.014.png)|
|**ROC Area under Curve** |
|0.76 |0.89 |0.93 |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.015.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.016.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.017.png)|
|**Testing Accuracy** |
|76.94 |88.99% |93.18% |

|<p>(penalty = norm 1, Algorithms </p><p>- liblinear, Inverse of </p><p>regularization strength = 1) </p>|(max\_depth = 45) |<p>(max\_depth = 45,n\_estimators </p><p>- 80) </p>|
| - | - | - |
|**AdaBoost** |**AdaBoost (alternative)** ||
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.018.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.019.png)||
|**ROC Area under Curve** |
|0.82 |0.93 ||
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.020.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.021.png)||
|**Testing accuracy** |
|81.74% |93.20% ||
|(base\_estimator = tree.DecisionTreeClassifier(ma x\_depth=2), n\_estimators= 1400, random\_state=0) |(base\_estimator = tree.DecisionTreeClassifier(ma x\_depth=23), n\_estimators=50, random\_state=0) ||
Comparing each Supervised Learning model, we get the best accuracy by the random forest  model.  Discussing  the  differences  between  the  decision  tree  model  and Random Forest, Decision trees spend less run time to build the model. As observing the ROC area under the curve, we can tell that Random forests have a larger area than a decision tree. Consistent with this result, the Random Forest has better performance than the Decision tree. According to the confusion matrix, both random forest and a decision tree predict rain tomorrow more accurately than predict not rain tomorrow. We expected the Adaboost model performance would work better than the random forest, however, the final result was the Adaboost couldn’t reach the same accuracy as the random forest. The reason is that the dataset is big enough to avoid overfitting for random forest models. We need to use a lot of stumps to reach higher accuracy, and it costs a lot of run time to build the model. Thus, we increase the max depth on Adaboost (change stump into a tree). The accuracy of the alternative Adaboost gets 93 percent accuracy which is close to the Random Forest result. 

Compared with our supervised learning and previous work, we can see that our model works better than a logistic regression model. The reason is that our model is much more complex than logistic regression. Although we need more time to build the model, we get better results. 

**Semi-supervised Learning:** 



|**LabelPropagation** |**LabelSpreading** |**Random Forest** |
| - | - | - |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.022.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.023.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.024.png)|
|0.77 |0.78 |0.93 |
|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.025.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.026.png)|![](Aspose.Words.5c48723f-c3ee-4cd9-aa23-1061cbbbf355.027.png)|
|77.85% |78.17% |93.18% |
|(kernel='knn', n\_neighbors= 20 , gamma=0, max\_iter=5000, tol=0.0001) |(kernel='knn', n\_neighbors= 20, gamma=0, alpha= 0.03 , max\_iter=5000, tol=0.0001) |<p>(max\_depth = 45,n\_estimators </p><p>- 80) </p>|
For the Semi-Supervised Learning, we want to try if we can get the same result as the supervised learning  model. To  simulate that we  have  lots  of  unlabeled  data, we removed some of the labels from our data. The result shows that the sklearn semi supervised model cannot handle data with so many features, both label propagation and label spreading algorithm get less than 80 percent of accuracy. Labelspreading got a  slightly  better  result because  the  alpha  parameter  can  specify  the  relative amount that an instance should adopt the information from its neighbors as opposed to its initial label. 

According to Supervised Learning and Semi-Supervised Learning, our best model will be the Random forest model with max depth equal 45 and n\_estimate equal 80 trees. The reason that the Semi-Supervised learning algorithm cannot achieve the same performance that random forest reaches is because we have enough data to get the perfect result on this binary problem. Thus, it is hard for Semi-Supervised learning to have equal or better results as Supervised Learning.  

5. Contributions of each team member 

In  this  project,  we  discuss  every  approach  together  and  we  write  our  code  on Collaboratory, which can code together and discuss together.  

6. Summary and conclusions 

In conclusion, for Supervised Learning problems, because we already have a lot of data, so the supervised learning model will come out with a result that is above average. For Semi-Supervised Learning, we can only get a result under 80%, but we think it is still usable when we get only a small bunch of data. If today we get a dataset that only has some data with a label, we think the Semi-Supervised Learning will outperform Supervised Learning. The Supervised Learning model can only learn from 

those labeled data, because those unlabeled data will not bring any benefits to the Supervised model. 

7. References  
1. "Rain in Australia" 03 December 2018. [Online]. Available: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package 
1. “Logistic Regression Classifier Tutorial with python” [Online]. Available: https://www.kaggle.com/prashant111/logistic-regression-classifier-tutorial 
1. “Iterative Imputer” [Online]. Available:  https://sklearn.apachecn.org/docs/master/41.html 
1. “Basic semi-supervised learning models”[Online]. Available: <https://www.kaggle.com/altprof/basic-semi-supervised-learning-models>
Page PAGE15 of NUMPAGES15 
