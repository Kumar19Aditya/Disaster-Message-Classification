import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
import pickle
import re
import base64
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Function to convert an image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Set the path to your local image
image_path = "Images\shutterstock-743541502.webp"
image_base64 = image_to_base64(image_path)

# Set a custom background and CSS styling
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{image_base64}"); 
    background-size: cover; 
    background-position: center; 
}}
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background for sidebar */
}}
h1, h3, h2 {{
    color: #000000;  /* Black color for headers */
    text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.7);  /* Light shadow for better visibility */
}}
.stTextArea textarea {{
    background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white background for text area */
    border: 1px solid #ccc;  /* Border styling */
}}
.stButton>button {{
    background-color: #FF5733;  /* Stylish button color */
    color: white;  /* Button text color */
    border: None;  /* No border */
    border-radius: 5px;  /* Rounded corners */
    padding: 10px 20px;  /* Padding for the button */
}}
.stButton>button:hover {{
    background-color: #C70039;  /* Darker shade on hover */
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# Function to tokenize text
def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Streamlit app
def main():
    st.title("Disaster Message Classification")

    # Sidebar for project sections
    st.sidebar.title("Project Overview")
    section = st.sidebar.selectbox("Select a section", 
                                     ["Data Collection", "Data Cleaning", "Model Training", "Model Evaluation", "Deployment"])

    # Fixed file paths for DataFrames
    file_path1 = r"Data/disaster_messages.csv"  # Replace with your actual file path
    file_path2 = r"Data/disaster_categories.csv"  # Replace with your actual file path

    # Display content based on the selected section in the sidebar
    if section == "Data Collection":
        st.sidebar.subheader("Data Collection")
        st.sidebar.write("In this project, two primary datasets are used for training and evaluating the model.")
        
        # Load and display DataFrame 1
        df1 = pd.read_csv(file_path1)
        st.sidebar.write("### Dataset 1: Disaster Messages")
        st.sidebar.write("This dataset contains various messages sent during disaster events, collected from different sources.")
        st.sidebar.write("#### Sample Data:")
        st.sidebar.dataframe(df1)

        # Column meanings for DataFrame 1
        st.sidebar.write("### Column Meanings:")
        st.sidebar.write("1. **ID**: Unique Identity of the Data")
        st.sidebar.write("3. **Message**: The actual message sent during a disaster event.")
        st.sidebar.write("3. **Original**: The message in native languages")
        st.sidebar.write("4. **Genre**: News/Direct/Social")
        
        # Load and display DataFrame 2
        df2 = pd.read_csv(file_path2)
        st.sidebar.write("### Dataset 2: Disaster Categories")
        st.sidebar.write("This dataset contains the classification of messages into various disaster-related categories.")
        st.sidebar.write("#### Sample Data:")
        st.sidebar.dataframe(df2)

        # Column meanings for DataFrame 2
        st.sidebar.write("### Column Meanings:")
        st.sidebar.write("1. **Message_ID**: A unique identifier for each message.")
        st.sidebar.write("2. **Category**: The category assigned to the message, indicating the type of disaster assistance needed.")

        # Data Collection Overview
        st.sidebar.write("### Data Collection Overview")
        data = pd.DataFrame({
        'Source': ['Dataset 1', 'Dataset 2'],
        'Count': [len(df1), len(df2)]
        })
        st.sidebar.bar_chart(data.set_index('Source'), color='#F7520B')  # Blue color

    elif section == "Data Cleaning":
        st.sidebar.subheader("Data Cleaning")
        st.sidebar.write("Explain how you cleaned the data, including any preprocessing steps.")
         # List specific data cleaning steps
        st.sidebar.write("### Data Cleaning Steps")
        st.sidebar.write("""
        1. **Handling Missing Values**: Checked for missing data and filled or removed missing entries as needed.
        2. **Outlier Detection and Removal**: Identified and handled outliers that could affect the model's performance.
        3. **Data Type Conversion**: Ensured that data types were correctly assigned (e.g., dates were converted to datetime format, categorical variables to category).
        4. **Encoding Categorical Variables**: Applied encoding techniques like one-hot encoding for categorical variables.
        5. **Data Normalization/Standardization**: Scaled the data to improve model performance where required.
        6. **Feature Engineering**: Created new features or modified existing features to better capture the relationships in the data.
        """)


    elif section == "Model Training":
        st.sidebar.subheader("Model Training")
        st.sidebar.write("This section provides an overview of the model training process, including the selected algorithms, data transformations, and any hyperparameter tuning applied.")

        # Describe the multi_tester function and the models used
        st.sidebar.write("### Model Training Details")
        st.sidebar.write("""
        The `multi_tester` function was created to streamline the training and evaluation of various models. Each model is fitted on the provided training data and evaluated to compare performance. The function performs the following steps:
        
        1. **Tokenization and Vectorization**: Each pipeline uses `CountVectorizer` with a custom tokenizer (`tokenize` function) to convert text data into a matrix of token counts. This is followed by the `TfidfTransformer` to apply TF-IDF (Term Frequency-Inverse Document Frequency) transformations for improved text feature representation.
        
        2. **Multi-Output Classifiers**: Since the problem involves multiple output labels, each classifier is wrapped in a `MultiOutputClassifier` to handle multi-label classification tasks.

        3. **Algorithms Used**:
        - **RandomForestClassifier**: A versatile ensemble model that aggregates multiple decision trees, aimed at improving accuracy and reducing overfitting.
        - **ExtraTreesClassifier**: Similar to RandomForest but builds extra randomized trees, increasing diversity and potentially improving performance in high-variance scenarios.
        - **GradientBoostingClassifier**: Uses gradient boosting on decision trees for better performance with complex relationships in the data.
        - **AdaBoostClassifier**: An ensemble method that combines weak learners sequentially, adjusting their weights to focus on difficult cases.
        - **Support Vector Classifier (SVC)**: A robust classifier for higher-dimensional data, effective in multi-label setups with the MultiOutputClassifier wrapper.
        
        4. **Pipeline Execution**: Each pipeline was fitted on the training data `X` and `y`, with model parameters displayed for transparency and tracking. 
        
        Each model's pipeline settings and transformations are detailed to ensure reproducibility.
        """)

        

    elif section == "Model Evaluation":
        st.sidebar.subheader("Model Evaluation")
        st.sidebar.write("Discuss how you evaluated the model's performance, including metrics used.")

    
        st.sidebar.write("### Model Performance Metrics")
        # (Metrics of models as before...)

        st.sidebar.write("### Best Performing Model")
        st.sidebar.write("""
        Based on the evaluation metrics, the **AdaBoostClassifier** is the best-performing model.
        It achieved the highest F1-score of 0.66 and a high recall of 0.58, making it the most balanced model in terms of precision and recall. 
        The other models, like **RandomForestClassifier** and **ExtraTreesClassifier**, had lower performance in comparison.

        ### Hyperparameter Tuning with AdaBoost
        After identifying AdaBoost as the best-performing model, I performed **hyperparameter tuning** to further improve its performance using **GridSearchCV**.
        
            The hyperparameters tuned are:
            - `tfidf__use_idf`: Whether to use inverse document frequency (True/False).
            - `clf__estimator__n_estimators`: The number of weak learners (trees) in the AdaBoost model, chosen between 50 and 100.
            - `clf__estimator__random_state`: The random state for reproducibility, set to 42.
            - `clf__estimator__learning_rate`: The learning rate to control the contribution of each weak learner, set to 0.5.
            
            Here is the code snippet for the **GridSearchCV** implementation:
            ```python
            parameters = {
                'tfidf__use_idf': (True, False),
                'clf__estimator__n_estimators': [50, 100], 
                'clf__estimator__random_state': [42],
                'clf__estimator__learning_rate': [0.5]
            }
            
            cv = GridSearchCV(pipeline, param_grid=parameters, cv=10,
                            refit=True, verbose=1, return_train_score=True, n_jobs=-1)
            ```

            The **GridSearchCV** performs a 10-fold cross-validation to find the best combination of hyperparameters that maximizes model performance.
            After the search, the best parameters and the corresponding model are selected for deployment.
            """)



        

    elif section == "Deployment":
        st.sidebar.subheader("Deployment")
        st.sidebar.write(" how we deployed the model for use.")
        st.sidebar.write("""
            ### Deployment Process
            After finalizing the model and hyperparameter tuning, I pushed the project to **GitHub** for version control and collaboration. 

            To deploy the model and make it accessible as a web application, I used **Streamlit Cloud**. Streamlit Cloud provides an easy way to deploy and share interactive data science applications.

            Here are the steps I followed for deployment:
            1. I pushed the project repository to GitHub.
            2. I created a new Streamlit app on Streamlit Cloud, linking it to the GitHub repository.
            3. Streamlit Cloud automatically detected the repository and deployed the app.
            4. The app was successfully deployed and is now accessible as a live website.

            The deployed application allows users to interact with the model and see predictions based on real-time inputs. The website is hosted on Streamlit Cloud, and users can access it through the following link: [Your Streamlit App URL]
            """)

    # Streamlit input form for message entry
    message_input = st.text_area("Enter a message", "We are at KF Marotiere 85, we have food and water shortage, please send food for us")

    # Load the model when the app starts
    model_path = r'Model\classifierAdaBoost.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Show the model prediction when the button is clicked
    if st.button("Classify Message"):
        test_text = [message_input]
        test = model.predict(test_text)
        
        # Drop unwanted columns from target labels
        engine = create_engine(r'sqlite:///Data_Cleaned\DisasterResponse.db')
        df = pd.read_sql_table("messages", con=engine)
        y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
        targs_drop = ['offer', 'security', 'infrastructure_related', 'tools', 
                      'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'fire', 'other_weather']
        y_min = y.copy()
        y_min.drop(targs_drop, axis=1, inplace=True)

        # Output the categories predicted by the model, line by line
        predicted_categories = y_min.columns.values[(test.flatten() == 1)]
        for category in predicted_categories:
            st.write(f"- {category}")

if __name__ == "__main__":
    main()
