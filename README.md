## InstructAV: Instruction Fine-tuning Large Language Models for Authorship Verification

A public repository containing datasets and code for the paper "InstructAV: Instruction Fine-tuning Large Language Models for Authorship Verification"

### Installation
> pip install -r code/requirements.txt

### Dataset
The dataset includes samples selected from the IMDB, Twitter, and Yelp datasets. 

Files labeled with 'rebuttal' correspond to datasets under the 'Classification setting', which comprise randomly selected samples that have not passed consistency verification. 

The remaining files are associated with the 'Classification and Explanation setting', representing samples where explanation labels have successfully undergone consistency verification.

### Code
