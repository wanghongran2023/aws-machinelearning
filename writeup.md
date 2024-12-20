# Project Introduction

This project is mainly about machine learning with AWS. It contains four parts:
 - (1) Distributed training in AWS SageMaker
 - (2) Multi-thread training on AWS EC2
 - (3) Lambda function for prediction
 - (4) Lambda concurrency and endpoint policy

# Distributed training in AWS SageMaker

In this part, I use a SageMaker notebook to run model training and deploy code. For the notebook instance type, I choose ml.t3.medium because all the training runs as a SageMaker training job, and the notebook instance only needs to trigger the process. Therefore, there is no so much requirements for the instance's performance, and the cheapest instance type should be sufficient for this task.<br/>
Regarding storage, although the data is temporarily stored on the instance, the data size is small, so 5 GB of storage is enough.<br/>
All the code and screenshots are stored in the **code** and **Training and Deployment** folders, and the descriptions are as follows:

| File Name | Description |
| --------- | ----------- |
| code/train_and_deploy-solution.ipynb | nodebook file for the end to end machine learning process |
| code/hpo_correct.py | python file for hyperparameter tuning |
| code/hpo_distributed.py | python file for sagemaker distributed training |
| code/inference2.py | entry point for sagemaker endpoint |
| Training and deployment/endpoint.png | screenshot for endpoint |
| Training and deployment/s3.png | screenshot for s3 |
| Training and deployment/nodebook.png | screenshot for nodebook instance |






