# ML
ML Classification problem

Exploratory Data Analysis: 
Found out that there are 60000 plus records and only one feature.  There are 15 classifiers that are highly unbalanced.  There were null values and those records were deleted as part of the data reprocessing.  Assuming the given data is accurate and reliable.
Models shortlisted for classification problem:  
1.	Naive Bayes with multiclass
Checked cross validation accuracy and selected alpha 0.01 and the accuracy score was 0.78.
2.	Linear SVM with pipeline 
Chose the pipeline for better performance and then ran the SGD classifier with penalty l2 and alpha 1e-3.  The F-score was 0.78 when the average set was weighted.   I chose the weighted option because the classifiers were unbalanced.
3.	Linear SVC with SMOTE 
I chose SMOTE to oversample the data so the classifier imbalance problem can be addressed.  The F-score was 0.85
4.	Gradient Boosting Classifiers and XGBoost
When running the above models, the system was not able to support it with the given memory and RAM.
Linear SVC with SMOTE took a long time to run, however since the accuracy score was the best, I used it to be my final model.  If the response time was more important than the accuracy, I would have used Linear SVM without SMOTE.
Front End:
The Flask framework was chosen to be deployed as a microservice on AWS as a platform using EC2  

Instructions: 
1)Download the folder as is.
2)Run the app.py and a URL will be generated. 
3)Click on the URL generated and you will get the UI(check screenshots folder:first.PNG). Enter the document content in the given text field.
4)You will be taken to a new page with the predicted class and confidence levels. 

Note : Current laptop configuration and state was very slow for me to execute the tasks.
While Deploying the web service, my AWS free credits expired and started billing. Hence I couldnt deploy it now. 

I tried Heroku. However, since my pickle file is too huge, github is unable to take that file. I tried zipping the file, even then it was unable to get into less than 25MB.
My next steps would have been: Uploading the file as a google drive link on AWS using curl commands. 
Note: Within the time and physical limitations I had, I couldn't proceed further.
The link for my pickle file : https://drive.google.com/file/d/16DhGgw6e0_6c6mxSlI6jHvjfIGZwzae5/view?usp=sharing


