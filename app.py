import streamlit as st
import sqlite3
import pandas as pd
import pickle
import base64

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def add_bg_from_local(image_file):

    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(
            image.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background:
            linear-gradient(
                rgba(0,0,0,0.45),
                rgba(0,0,0,0.45)
            ),
            url("data:image/jpg;base64,{encoded_string}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local("360_F_292169498_87G9Ue6DVTcUO9XNw6H39DgwjAegicEg.jpg")

# --------------------------
# DATABASE CONNECTION
# --------------------------

conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()

# --------------------------
# SESSION STATE
# --------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "model" not in st.session_state:
    st.session_state.model = None
    
if "quiz_page" not in st.session_state:
    st.session_state.quiz_page = False

if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

# --------------------------
# SIDEBAR MENU
# --------------------------
page = st.sidebar.selectbox(
    "Menu",
    ["Register", "Login"]
)

# ==========================
# SIDEBAR QUIZ GAME
# ==========================
st.sidebar.markdown("---")

if st.sidebar.button("🎮 Start Quiz"):
    st.session_state.quiz_page = True
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.rerun()

questions = [
    {
        "question": "IBM stands for?",
        "options": [
            "International Business Machines",
            "Indian Business Machines",
            "International Banking Market",
            "Integrated Business Model"
        ],
        "answer": "International Business Machines"
    },
    {
        "question": "Which ML algorithm is commonly used for classification?",
        "options": [
            "Catboost",
            "K-Means",
            "Apriori",
            "PCA"
        ],
        "answer": "Catboost"
    },
    {
        "question": "What does CSV stand for?",
        "options": [
            "Comma Separated Values",
            "Computer Stored Values",
            "Column Sorted Values",
            "Common Structured Variables"
        ],
        "answer": "Comma Separated Values"
    },
    {
        "question": "Attrition means?",
        "options": [
            "Employee Leaving Company",
            "Employee Promotion",
            "Employee Salary",
            "Employee Attendance"
        ],
        "answer": "Employee Leaving Company"
    },
    {
        "question": "Which library is used for DataFrames in Python?",
        "options": [
            "Pandas",
            "NumPy",
            "Matplotlib",
            "TensorFlow"
        ],
        "answer": "Pandas"
    }
]

if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if st.session_state.quiz_index < len(questions):

    q = questions[st.session_state.quiz_index]

    st.sidebar.write(
        f"Question {st.session_state.quiz_index + 1} of {len(questions)}"
    )

    selected = st.sidebar.radio(
        q["question"],
        q["options"],
        key=f"quiz_{st.session_state.quiz_index}"
    )

    if st.sidebar.button("Submit Answer"):

        if selected == q["answer"]:

            st.sidebar.success(
                "🎉 Correct! You're smarter than the HR manager!"
            )

            st.session_state.quiz_score += 1
            st.session_state.quiz_index += 1

            st.rerun()

        else:

            funny_messages = [
                "😂 Wrong! Even the office coffee machine knew that.",
                "🤦 Oops! HR is taking notes on this answer.",
                "🤣 That's not it! The interns are laughing.",
                "🙈 Wrong answer! Try not to put that on your resume.",
                "😅 Nice try! The dataset disagrees."
            ]

            import random

            st.sidebar.error(
                random.choice(funny_messages)
            )

else:

    st.sidebar.success(
        f"🏆 Quiz Completed! Score: {st.session_state.quiz_score}/5"
    )

    if st.sidebar.button("Play Again"):

        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0

        st.rerun()

if st.session_state.quiz_page:

    st.markdown(
    """
    <h1 style="
    text-align:center;
    color:#FFD700;
    font-size:55px;
    font-family:Georgia;
    text-shadow:3px 3px 5px black;">
    🎮 HR QUIZ CHALLENGE
    </h1>
    """,
    unsafe_allow_html=True

    )

    q_no = st.session_state.quiz_index

    if q_no < len(questions):

        q = questions[q_no]

        st.markdown(
    f"""
    <h2 style="
    color:white;
    font-size:35px;
    text-align:center;">
    Question {q_no+1} of {len(questions)}
    </h2>
    """,
    unsafe_allow_html=True
)

        answer = st.radio(
            q["question"],
            q["options"]
        )

        if st.button("Submit"):

            if answer == q["answer"]:

                st.success(
                    "🎉 Correct! Moving to next level..."
                )

                st.session_state.quiz_score += 1
                st.session_state.quiz_index += 1

                st.rerun()

            else:

                import random

                funny = [

                    "😂 Wrong! Even the coffee machine knew that.",

                    "🤣 HR has filed a complaint against this answer.",

                    "🙈 Oops! The interns are laughing.",

                    "🤦 Wrong answer! Try again future data scientist.",

                    "😆 The dataset rejected your answer."
                ]

                st.error(
                    random.choice(funny)
                )

    else:

        st.balloons()

        st.success(
            f"🏆 Quiz Completed!\n\nScore: {st.session_state.quiz_score}/5"
        )

        if st.button("Back To Main Page"):

            st.session_state.quiz_page = False
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0

            st.rerun()

    st.stop()

# --------------------------
# REGISTER PAGE
# --------------------------

if page == "Register":

    st.title("User Registration")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Save"):

        if username == "" or password == "":
            st.warning("Please enter username and password")

        else:

            try:

                cursor.execute(
                    "INSERT INTO users(username,password) VALUES(?,?)",
                    (username, password)
                )

                conn.commit()

                st.success(
                    "Registration Successful"
                )

                st.info(
                    "Go to Login from Sidebar"
                )

            except sqlite3.IntegrityError:

                st.error(
                    "Username Already Exists"
                )

# --------------------------
# LOGIN PAGE
# --------------------------

elif page == "Login":

    st.title(
        "IBM HR EMPLOYEE ATTRITION DATASET BROWSER"
    )

    if not st.session_state.logged_in:

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            cursor.execute(
                """
                SELECT * FROM users
                WHERE username=? AND password=?
                """,
                (username, password)
            )

            user = cursor.fetchone()

            if user:

                st.session_state.logged_in = True
                st.session_state.username = username

                st.success(
                    "Login Successful"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid Username or Password"
                )

    else:

        st.success(
            f"Welcome {st.session_state.username}"
        )

        if st.button("Logout"):

            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.model = None

            st.rerun()

        # --------------------------
        # CSV FILE UPLOAD
        # --------------------------

        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                df = pd.read_csv(
                    uploaded_file
                )

                # --------------------------
                # DATASET PREVIEW
                # --------------------------

                st.subheader("First 10 Rows")
                st.dataframe(df.head(10))

                st.subheader("Last 10 Rows")
                st.dataframe(df.tail(10))

                # --------------------------
                # SHAPE
                # --------------------------

                st.subheader("Dataset Shape")

                st.write(
                    f"Rows : {df.shape[0]}"
                )

                st.write(
                    f"Columns : {df.shape[1]}"
                )

                # --------------------------
                # COLUMN NAMES
                # --------------------------

                st.subheader("Column Names")

                st.write(
                    df.columns.tolist()
                )

                # --------------------------
                # DATA TYPES
                # --------------------------

                st.subheader("Data Types")

                st.dataframe(
                    df.dtypes.astype(str)
                )

                # --------------------------
                # MISSING VALUES
                # --------------------------

                st.subheader(
                    "Missing Values"
                )

                st.dataframe(
                    pd.DataFrame(
                        df.isnull().sum(),
                        columns=[
                            "Missing Count"
                        ]
                    )
                )

                # --------------------------
                # DUPLICATE ROWS
                # --------------------------

                st.subheader(
                    "Duplicate Rows Count"
                )

                st.write(
                    df.duplicated().sum()
                )

                duplicates = df[
                    df.duplicated()
                ]

                st.subheader(
                    "Duplicate Records"
                )

                if len(duplicates) > 0:

                    st.dataframe(
                        duplicates
                    )

                else:

                    st.success(
                        "No Duplicate Records Found"
                    )

                # --------------------------
                # UNIQUE VALUES
                # --------------------------

                st.subheader(
                    "Unique Values Count"
                )

                st.dataframe(
                    pd.DataFrame(
                        df.nunique(),
                        columns=[
                            "Unique Count"
                        ]
                    )
                )

                # --------------------------
                # LABEL ENCODING
                # --------------------------

                encoded_df = df.copy()

                encoders = {}

                for col in encoded_df.columns:

                    if encoded_df[col].dtype == "object":

                        le = LabelEncoder()

                        encoded_df[col] = (
                            le.fit_transform(
                                encoded_df[col].astype(str)
                            )
                        )

                        encoders[col] = le

                st.subheader(
                    "Label Encoded Dataset"
                )

                st.dataframe(
                    encoded_df.head(10)
                )

                # --------------------------
                # TARGET COLUMN
                # --------------------------

                target = st.selectbox(
                    "Select Target Column",
                    encoded_df.columns
                )

                st.success(
                    f"Selected Target : {target}"
                )

                # --------------------------
                # MODEL FILE UPLOAD
                # --------------------------

                st.subheader(
                    "Upload Trained Model (.pkl)"
                )

                model_file = st.file_uploader(
                    "Choose Model File",
                    type=["pkl"]
                )

                if model_file is not None:

                    try:

                        model = pickle.load(
                            model_file
                        )

                        st.success(
                            "Model Loaded Successfully"
                        )

                        X = encoded_df.drop(
                            columns=[target]
                        )

                        y = encoded_df[target]

                        predictions = model.predict(X)

                        accuracy = accuracy_score(
                            y,
                            predictions
                        )

                        st.success(
                            f"Model Accuracy = {accuracy*100:.2f}%"
                        )

                        st.session_state.model = model
                        st.session_state.features = X.columns.tolist()
                        st.session_state.target = target
                        st.session_state.df = encoded_df

                    except Exception as e:

                        st.error(
                            f"Model Error : {e}"
                        )

                # --------------------------
                # PREDICTION SECTION
                # --------------------------

                if st.session_state.model is not None:

                    st.subheader(
                        "Predict New Employee Data"
                    )

                    user_input = {}

                    for col in st.session_state.features:

                        default_value = float(
                            st.session_state.df[col].mean()
                        )

                        user_input[col] = st.number_input(
                            f"Enter {col}",
                            value=default_value
                        )

                    if st.button("Predict"):

                        input_df = pd.DataFrame(
                            [user_input]
                        )

                        prediction = (
                            st.session_state.model.predict(
                                input_df
                            )
                        )

                        st.subheader(
                            "Prediction Result"
                        )

                        if target.lower() == "attrition":

                            if prediction[0] == 1:

                                st.error(
                                    "Attrition = YES"
                                )

                            else:

                                st.success(
                                    "Attrition = NO"
                                )

                        else:

                            st.success(
                                f"Predicted {target}: {prediction[0]}"
                            )

            except Exception as e:

                st.error(
                    f"Dataset Error : {e}"
                )