# Fake-News-detection
Modeling Fake News detection problem using Content-based and Social Context data

This readme.txt file contains notes regarding the Python 3 code
and sample data which accompany my Master Thesis:

“Fake News Detection using social context and textual data”,
Evangelia Grigoria ZVE
Master’s degree (2nd year) Statistics and Econometrics (Distance Learning)
Toulouse School of Economics / Université Toulouse Capitole 1
Academic Year: 2018/2019

The Python 3 code file, and sample data files, are described
below.

This directory includes one fake news dataset that contain both the news contents and social context information.

News directory :
      training: this directory contains the content of the news in the training set. Each file is a news stored in the txt format and contains the title, the summary and the content of the news. The name of the txt file is @news_id.txt where @news_id is the identifier of the news.
      test: this directory contains the content of the news in the test set. The news are represented in the same format than for the training directory

newsUser.txt: the news-user relationship. For example, '240 1 1' means news 240 is posted/spreaded by user 1 for 1 time.

UserUser.txt: the user-user relationship. For example, '1589 1' means user 1589 is following user 1.

labels_training.txt: indicate whether the news in the training set is fake (1) or real (0). For example '23 0' means the news 23 is real.

labels_test.txt: indicate whether the news in the test set is fake (1) or real (0). For example '23 0' means the news 23 is real.

In the "Loading Datasets" part of the code, which is situated at the beggining of the file, you should replace the file paths with the ones of your computer (line 81 to line 98).

The code follows a logic continuity based on my work process, chapters and results in the manuscript.
It can be exectuted as a unique file.

When the code is related to a specific Chapter this is indicated before the code starts with comments.

For example, you will see a comment "#RELATED CHAPTER IN MANUSCRIPT: 3.2.3 News Propagation - General measures",
that means the code below that title concerns the results and plots presented on that Chapter and Subchapters, until a next title with this mention is presented.

There are parts of the code that concern data preparation task, and thus you will not see "RELATED CHAPTER IN MANUSCRIPT:" mention but only regular comments.

From the beggining to the end of the file, in the code you can find comments that can guide you to understand to which result each par of the code is associated.


