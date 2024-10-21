News Documents Retrieval(State and National News)

This is a retrieval process system that allows users to search for relevant news articles.

I scrapped the news articles dataset from www.kaggle.com 

-------------------------------------------------------------------------------

You may find the zipped dataset and json files folder in the below drive link
https://drive.google.com/drive/folders/1X-r9QC294JSDh9dBOtBBDcxs2bzjmCb2


Note: Please Download the  folder named "Dataset".

Please download "Dataset" folder and store it in Code folder .


In Code folder, there's a code named root.py and run the code you will get a new folder named "root".

And in  "Ranking.py" code,place the path of the Dataset folder.
In "ranking.py" code, root gets automatically rendered no need to give path.
------------------------------------------------------------------------------
indexing for documents is already done and stored in "Dataset folder".

As Code folder contains root.py,ranking.py .

after giving the dataset path to ranking.py ,run the code. 

-------------------------------------------------------------------------------
1.After running the code you get a question to "enter a query" .
2.And after entering the query you will find content .
3.Next you type whether the document is relevant(1) and non-relevant(0).        
4.After typing all the feedback values.
5.all the precision and recall values will get printed.
6.and PR-curve and 11-interpolated curve will get as output
