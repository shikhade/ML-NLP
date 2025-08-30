# Python for Data Science: Foundations Project

This repository serves as a showcase of fundamental skills in Python for data science, covering core libraries like **NumPy**, **Pandas**, and **Scikit-learn**. The project is structured into three main modules, each designed to tackle common challenges in data manipulation, analysis, and machine learning.

---

## Module 1: NumPy Mastery 🧮

This module demonstrates proficiency in manipulating multi-dimensional arrays and implementing custom algorithms using NumPy. The focus is on efficient, vectorized operations.

### Key Implementations:
* **Multi-dimensional Array Operations**: A 5x4 array with random integers was created to perform tasks such as:
    * Extracting the **anti-diagonal** elements.
    * Computing **row-wise maximum values**.
    * Filtering elements that are less than or equal to the array's **mean**.
* **Algorithmic Traversal**: A function that traverses the boundaries of a 2D array in a **clockwise** direction, returning a 1D array of the boundary elements.
* **1D Array Logic**: A 1D array of 20 random floats was generated to showcase:
    * Statistical computations (**min, max, median**).
    * Conditional transformations (e.g., **squaring elements** less than 5).
    * A custom sorting function to arrange elements in an **alternating smallest-largest pattern** (e.g., `[min, max, 2nd_min, 2nd_max, ...]`).

---

## Module 2: Student Performance Analysis with Pandas 📊

This module simulates a practical data analysis task using Pandas. A dataset for 10 students across 10 subjects was generated from scratch to perform a complete analysis workflow.

### Analysis Workflow:
1.  **Data Creation**: A DataFrame was constructed to hold student grades.
2.  **Feature Engineering**: New columns were computed:
    * `total_score`: The sum of scores across all subjects.
    * `scaled_score`: The total score normalized to a 100-point scale.
    * `grade`: A letter grade (A, B, C, D, F) assigned based on the scaled score.
3.  **Data Aggregation & Filtering**:
    * Students were **sorted** by their `scaled_score` to rank their performance.
    * **Subject-wise average scores** were calculated to identify difficult subjects.
    * The dataset was **filtered** to show only high-performing students (grades 'A' or 'B').

### Sample Output:
| student_id | total_score | scaled_score | grade |
|:-----------|:------------|:-------------|:------|
| 8          | 875         | 87.5         | A     |
| 3          | 821         | 82.1         | A     |
| 5          | 789         | 78.9         | B     |
| ...        | ...         | ...          | ...   |

---

## Module 3: NLP for Sentiment Classification 🧠

This section demonstrates the fundamentals of Natural Language Processing (NLP) by building and comparing two different sentiment analysis models using Scikit-learn.

### Pipeline 1: Movie Review Analysis with Naive Bayes
* **Dataset**: 100 sample movie reviews (50 positive, 50 negative).
* **Vectorization**: Text was converted into numerical features using `CountVectorizer`, configured to use a maximum of 500 features and remove English stop words.
* **Modeling**: A `Multinomial Naive Bayes` classifier was trained on the vectorized data.
* **Evaluation**: The model achieved a **test accuracy of 88%**. A prediction function was built to classify new, unseen reviews.

### Pipeline 2: Product Feedback Classification with Logistic Regression
* **Dataset**: 100 sample product reviews (good/bad).
* **Vectorization**: Text was processed using `TfidfVectorizer`, which accounts for term frequency-inverse document frequency, helping to highlight more important words.
* **Modeling**: A `Logistic Regression` classifier was trained to distinguish between good and bad feedback.
* **Evaluation**: The model's performance was measured using **precision, recall, and F1-score**, providing a more nuanced view of its effectiveness than accuracy alone.

#### Example Prediction Function:
```python
# Function to classify new feedback
new_feedback = "The battery life on this is amazing and lasts all day!"
prediction = predict_product_sentiment(new_feedback)
print(f"Sentiment: {prediction}")
# Expected Output: Sentiment: Good
