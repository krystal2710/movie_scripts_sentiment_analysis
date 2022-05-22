# Sentiment Scores of Movie Scripts and <br> Their Performance on Comedy Classification Task

The project first aims at getting the information of what sentiments movies imply from the movie scripts to a decent level of accuracy. It then uses these sentiment scores for comedy classification task in comparison with using TF-IDF scores extracted from the movie scripts. The goal is to see whether sentiment scores help with better classifying comedy and reducing space and time complexity thanks to a smaller training dataset (sentiment scores) compared to the previous larger training dataset (TF-IDF scores).

## Data
- 644 movie scripts used in the project are extracted from the [Internet Movie Script Database (IMSDb) website](https://imsdb.com).
- The metadata of the corresponding movies is extracted from several datasets on [IMDb Datasets website](https://www.imdb.com/interfaces/).

## Repository Overview
| File name | Description |
| --------- | ----------- |
| README.md | this file   |    
| web_scraping.ipynb | code for web-scraping the movie scripts from the [IMSDb website](https://imsdb.com) |
| scripts | a folder containing all movie scripts' text files |   
| central_computation.ipynb | code for extracting sentiment scores and TF-IDF scores from movie scripts and training machine learning models on these scores |
| report.md | a written report summarizing the details of the project following the IMRAD template |
| image | a folder containing all images used in the report |
| metadata.csv | metadata of all movie scripts |
| metadata_labeled_30.csv | metadata of 30 labeled movie scripts |

## Getting Started
1. Clone this repo
2. For the web-scraping part, open `web-scraping.ipynb` and run the source code.
3. For the main part (extracting sentiment scores and TF-IDF scores from movie scripts and training machine learning models on these scores), open `central_computation.ipynb` and run the source code.
4. Read the final report, `report.md` for a detailed summary and discussion of the project.

## License
Distributed under the MIT License.


