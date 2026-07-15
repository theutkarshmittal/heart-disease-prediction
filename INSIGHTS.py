'''📊 Insight 1: Model Accuracy Comparison

📌 Observation
Random Forest achieved the highest test accuracy (89.1%).
KNN and SVM performed similarly with an accuracy of approximately 87%.
Logistic Regression and Naive Bayes also showed competitive performance.
Decision Tree recorded the lowest accuracy (81.5%).

💡 Business Insight:- 
Random Forest performed best on the test dataset,
correctly classifying approximately 89 out of every 100 patients.
However, since this evaluation is based on a single train-test split, 
additional validation is required before selecting it as the final deployment model.'''


'''📊 Insight 2: Cross Validation Comparison

📌 Observation
SVM achieved the highest average cross-validation accuracy (87.6%).
KNN and Logistic Regression followed closely.
Random Forest ranked fourth despite having the highest test accuracy.
Decision Tree remained the weakest performer.

💡 Business Insight
Cross-validation indicates that SVM generalizes better to unseen patient data than
the other models. Although Random Forest achieved the highest single-test accuracy,
SVM demonstrated more reliable performance across multiple data splits, making it a 
stronger candidate for deployment.'''

'''📊 Insight 3: Model Stability

📌 Observation
   Model                Std Dev
Naive Bayes	             0.017
Random Forest	         0.018
Logistic Regression	     0.019
SVM	                     0.025
KNN	                     0.027
Decision Tree	         0.035

💡 Business Insight
Lower standard deviation indicates that a model's performance is more consistent 
across different training-validation splits. Naive Bayes, Random Forest, 
and Logistic Regression exhibited the most stable performance,
while Decision Tree showed the highest variability, indicating greater sensitivity 
to changes in the training data.'''

'''📊 Overall Project Conclusion

Six supervised machine learning algorithms were evaluated for heart disease prediction. 
While Random Forest achieved the highest accuracy on the test dataset (89.1%), 
cross-validation demonstrated that SVM achieved the highest average accuracy (87.6%),
suggesting superior generalization to unseen data. Decision Tree consistently showed 
the lowest performance and the highest variability. Based on the evaluation results,
SVM is currently the preferred model for deployment, while Random Forest remains 
a strong alternative.'''


'''Key Takeaways :==

🌲 Random Forest achieved the highest single test accuracy (89.1%).
🎯 SVM achieved the highest cross-validation accuracy (87.6%), indicating the strongest generalization capability.
📈 Logistic Regression and Naive Bayes showed stable performance with relatively low variability.
⚠️ Decision Tree was the least reliable model due to both lower accuracy and higher variation across folds.
🚀 Based on the current evaluation, SVM is selected for deployment,
    with Random Forest identified as a strong candidate for future optimization
    through hyperparameter tuning.'''

