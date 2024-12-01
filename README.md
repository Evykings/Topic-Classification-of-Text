# 📚 Text Classification Project  
A Data Science project leveraging Machine Learning and Natural Language Processing (NLP) techniques to classify textual data into predefined categories. This project demonstrates end-to-end workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## 📝 Project Overview  
This project aims in leveraging machine learning to analyze and classify paragraph-sized text submissions into specific topics. By automating this process, the organization can better categorize content and enhance user experience on their platform. This project explores a robust approach to text classification by applying machine learning algorithms to process and categorize textual data efficiently.

### Key Objectives:  
- **Preprocessing textual data**: Tokenization, stop-word removal, lemmatization.  
- **Feature extraction**: Leveraging TF-IDF and word embeddings for vector representation.  
- **Modeling**: Training machine learning classifiers (SGDClassifier and MultinomialNB).  
- **Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.  

---

## 📊 Dataset Description  
The dataset provided for this project contains the following features, each contributing unique insights for text classification. Below is a brief description of each feature:.

| Feature Name        | Description                                                                                      |    
|---------------------|--------------------------------------------------------------------------------------------------|
| par_id              | A unique identifier for each paragraph to be classified.                                         | 
| paragraph           | The main textual content to classify, varying in length and complexity.                          | 
| has_entity          | Binary indicators specifying whether the paragraph references:                                   | 
|                     | - Product (```yes/no```)                                                                         | 
|                     | - Organisation (```yes/no```)                                                                    |
|                     | - Person (```yes/no```)                                                                          |
| lexicon_count       | The total number of words in the paragraph, providing an idea of its length.                     |
| difficult_words     | The number of challenging words in the text, based on a predefined lexicon of difficult terms.   |
| last_editor_gender  | The gender of the last person who edited the paragraph, offering potential demographic insights. |
| category            | The target classification label for the text, representing one of the five specific topics:      |
|                     | - Artificial Intelligence                                                                        |
|                     | - Movies about Artificial Intelligence                                                           |
|                     | - Programming                                                                                    |
|                     | - Philosophy                                                                                     |
|                     | - Biographies                                                                                    |
| text_clarity        | A qualitative measure of the text's clarity level; initially sparse in labeled data.             | 

### Dataset Features:  
- **Text**: The primary input data to be classified.  
- **Label**: The target variable indicating the class of each text.  
  

### Dataset Statistics:  
- **Number of Samples**: [ 9347 records]  
- **Number of Classes**: [ 5 categories]  
- **Class Distribution**: Visualized in the EDA section below.  

---

## 🔍 Exploratory Data Analysis (EDA)  
- Explored text length distributions, word frequency analysis, and class distributions.  
- Visualized insights using and bar plots.
![alt text](https://github.com/Evykings/Topic-Classification-of-Text/blob/main/distribution%20of%20category.png)
![alt text](https://github.com/Evykings/Topic-Classification-of-Text/blob/main/distribution%20of%20gender.png)
![alt text](https://github.com/Evykings/Topic-Classification-of-Text/blob/main/tc.png)

---

## ⚙️ Data Preprocessing  
1. **Text Cleaning**:  
   - Removed special characters, punctuations, and HTML tags.  
   - Converted text to lowercase for consistency.  

2. **Tokenization**:  
   - Split text into individual words or tokens.  

3. **Stop-Word Removal**:  
   - Removed commonly used words with little contextual meaning (e.g., "the," "and").  

4. **Stemming/Lemmatization**:  
   - Reduced words to their base or root form.  

5. **Vectorization**:  
   - Implemented **TF-IDF** for traditional models.  
   - Used **word embeddings** (e.g., Word2Vec, GloVe) for deep learning approaches.

---

## 🧠 Model Building  
Implemented and evaluated multiple models:  

1. **Baseline Models**:  
   - Random Trivial Baseline.  

2. **Advanced Models**:  
   - SGDClassifier and MultinomialNB.  

### 🔬 SGDClassifier Model Performance :  
| Categories                            | Precision | Recall | F1-Score |  
|---------------------------------------|-----------|--------|----------|  
| artificial intelligence               |  0.90     | 0.93   | 0.91     |  
| biographies                           |  0.94     | 0.92   | 0.93     |  
| movies about artificial intelligence  |  0.93     | 1.00   | 0.97     |
| philosophy                            |  0.93     | 0.91   | 0.92     |
| programming                           |  0.95     | 0.99   | 0.97     |

### Confusion Matrix
![alt text](https://github.com/Evykings/Topic-Classification-of-Text/blob/main/confusion%20matrix%20for%20sgd.png)


### 🔬 MultinomialNB Model Performance :  
| Categories                            | Precision | Recall | F1-Score |  
|---------------------------------------|-----------|--------|----------|  
| artificial intelligence               |  0.89     | 0.92   | 0.90     |  
| biographies                           |  0.92     | 0.89   | 0.91     |  
| movies about artificial intelligence  |  0.96     | 0.96   | 0.96     |
| philosophy                            |  0.90     | 0.90   | 0.90     |
| programming                           |  0.95     | 0.97   | 0.96     |

### Confusion Matrix
![alt text](https://github.com/Evykings/Topic-Classification-of-Text/blob/main/confusion%20matrix%20for%20MultinomialNB.png)

---
 
