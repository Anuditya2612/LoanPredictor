**Welcome to the LOAN Predictor application**

This is a multi-page web-application built using python with its libraries like pandas, matplotlib and scikit-learn, and then trained on Supervised learning techniques- Logistic Regression and RandomForest Classification in this case.

I used Kaagle's loan predictor dataset, comprising various features like Income, Size of Family, Education level, Access to CD/securities account(s) etc. and we perform necessary cleaning and pre-processing on the input features.
We take a look into the dataset including some visualizations like a countplot for target variables, and one for Education levels too, plus we manually assign Education labels to the integral feature.
Then on the basis of trained data (and normalized using StandardScaler), we make predictions on the test data using our models, and also draw comparison between the performance of the 2 models using evaluation metrics.

On the basis of input features entered by the users enabling user-interaction, the application gives the verdict ('Yes'/'No') along with a probability score whether the customer is likely to accept or reject the loan offer made by Banks, agencies etc., and thus provides business value to them as they can make their loan offers to specific customers, who are more likelier to accept the same.

Not only does it help adding up new customers for them but helps save time, and a lot of money spent on marketing and paying to people appointed specifically for this purpose on ground.

**About the key features**

From the feature importance plot we visualized in the RandomForest Classifier, we infer that 'Income' (in k$), 'CCAvg.' (again in k$) and 'Education' (labeled manually) are the most important ones, when decisions are made in a leaf node while we decide the no. of estimators/neighbors to be considered.
Additionally, 'Age' and 'CD Account' are important features too, as reflected from one of the tree of random forest we visualize.

**Challenges**

I faced problems in creating a virtual environment in VS Code, and resorted to some help from chatgpt. But after that, i was able to write the code (.ipynb) and the page codes (.py files) on my own, and then had to look up some sources including the blog on how we setup the main.py page.
But after i was done with it, i faced another problem while making a commit on github, as i changed the name of the folder and that broke some paths in the process. Then I had to setup a .venv again, install the requisite packages and then make the code run using the workspace interpreter.

After the commit was complete, it was easy-peasy with creating an app to deploy on stremlit, entering the repo name and finally deploying it live.
**Thus, i was able to deploy my first multi-page web-app live.**
