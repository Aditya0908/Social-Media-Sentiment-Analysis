# Social Media Sentiment Analysis

### Overview

Social Media Sentiment Analysis Model is a project focused on predicting sentiments (negative, positive, neutral) from social media captions. Leveraging Natural Language Processing (NLP) and machine learning techniques, the model aims to provide insights into the sentiment conveyed in textual content.

---

### Project Structure

- **Data Preparation:**
  - The project begins with importing and preprocessing data from the 'LabeledText.xlsx' file, extracting captions and corresponding sentiment labels.

- **Text Preprocessing:**
  - Textual data undergoes thorough preprocessing, including URL removal, tokenization, lowercasing, punctuation elimination, stop word removal, and stemming.

- **Feature Engineering:**
  - TF-IDF vectorization is applied to convert processed captions into numerical features suitable for machine learning models.

- **Model Training:**
  - Three models are explored: Multinomial Naive Bayes, Random Forest Classifier, and Logistic Regression. Hyperparameter tuning is performed using GridSearchCV to optimize model performance.

- **Model Evaluation:**
  - The models are evaluated based on accuracy scores, and the best-performing Logistic Regression model is selected.

- **Pipeline Creation:**
  - The project provides a user-friendly pipeline for future sentiment analysis. The preprocessing function and the trained Logistic Regression model are saved as pickle files for seamless integration into other applications or platforms.

---

### Files Included

1. **Sentiment_Model_kaggle_logistic_regression.pkl:**
   - The trained Logistic Regression model saved for future use.

2. **preprocess_model.pkl:**
   - The preprocessing function saved for consistent text preprocessing.

3. **tfidf_vectorizer.pkl:**
   - The TF-IDF vectorizer used for feature engineering.

4. **SentimentAnalysis.ipynb:**
   - Jupyter Notebook containing the entire codebase and detailed explanations.

5. **LabeledText.xlsx:**
   - The original dataset containing captions and corresponding sentiment labels.

---

### Usage

1. **Training the Model:**
   - Run the code in the 'SentimentAnalysis.ipynb' notebook to train and evaluate the sentiment analysis model.

2. **Pipeline Integration:**
   - Load the saved preprocessing function and the Logistic Regression model into your application using the provided pickle files for real-time sentiment prediction.

---

### Dependencies

- pandas
- numpy
- nltk
- scikit-learn
- pickle

Ensure the mentioned dependencies are installed before running the code.

---

### Contribution

Contributions are welcome! Feel free to submit issues or pull requests to enhance the functionality or address any improvements.

---

### License

This project is licensed under the [MIT License](LICENSE).

--- 

Feel free to reach out for any questions or clarifications.

Happy Sentiment Analysis! 🌾🤖
