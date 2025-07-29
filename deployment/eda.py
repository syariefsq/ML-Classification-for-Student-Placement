import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import numpy as np

def run():
    # --- Page Title and Header ---
    st.title("🎓 University Job Placement Prediction: Exploratory Data Analysis")
    st.markdown("""
    <style>
    .big-font {font-size:22px !important; font-weight:600;}
    .section-title {margin-top: 2em; margin-bottom: 0.5em;}
    .insight {background-color: #f0f4f8; border-left: 4px solid #1f77b4; padding: 0.7em 1em; margin: 1em 0; border-radius: 6px;}
    </style>
    """, unsafe_allow_html=True)

    st.image("https://media.tenor.com/87eweXpqvD0AAAAC/university-yay.gif", caption="Exploratory Data Analysis (EDA) in Progress", use_column_width=True)

    st.markdown('<div class="big-font">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('''
    An University in India with an MBA Program aims to maximize graduate job placement and effectively support their students to excel in their career. However, they face a problem with a lack of a data-driven understanding of how specific academic and non-academic factors can truly influence their placement success. Without these insights, it will be hard to identify tailored advice for the students and use internal resources from the university to boost overall placement numbers. Creating a disruptive prediction tool using the students' past information available at the university will help to assess the likelihood of getting a job placement. The main goal is to increase the employability of the MBA students of the university.
    ''')

    st.markdown('<div class="big-font">Problem Statement</div>', unsafe_allow_html=True)
    st.markdown('''
    The University with MBA Program struggles to improve graduate job placement and wants to find a way to support students to increase their employability.
    ''')

    st.markdown('---')
    st.markdown('<div class="big-font section-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)

    # 1. What does the dataset look like?
    st.markdown('### 1. What does the dataset look like?')
    st.markdown('''We begin by exploring the structure and contents of the dataset, which contains academic and non-academic information of MBA students, including their placement status.\n\n**Dataset Description:**\n- 215 MBA students\n- Academic performance, demographic info, work experience, and placement status\n\n**Context:**\nThis dataset is based on internal campus data from a university in India offering an MBA program. The aim is to understand the various academic and non-academic factors that contribute to a student's success in securing a job placement after graduation.\n\n**Columns include:**\n- Academic Performance: Secondary, higher secondary, degree, entrance test, and MBA percentages\n- Demographic: Gender, school/board, specialization, degree type\n- Work Experience: Prior work experience\n- Placement Status: Placed/Not Placed (target)\n- Salary: If placed\n''')
    df = pd.read_csv("./src/Placement_Data_Viz.csv")
    st.dataframe(df, use_container_width=True, height=350)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>The dataset provides a comprehensive view of student backgrounds, academic performance, and placement outcomes, which are essential for further analysis.</li>\n<li>Missing values are present in the <b>Salary</b> column, corresponding to students who were not placed. This will be handled later.</li>\n<li>Columns have been renamed for clarity and consistency, and unused columns (like index) have been dropped.</li>\n<li>Initial checks show the data types are appropriate and cardinality of categorical features is low, making them suitable for one-hot encoding.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*Having understood the overall structure and context of our dataset, the next logical step is to break down the student population by their categorical characteristics. This will help us identify any dominant groups or imbalances that could influence placement outcomes.*''')
    st.markdown('---')
    # 2. What are the distributions of categorical features?
    st.markdown('### 2. What is the distribution of students across different categorical features?')
    st.markdown('''First, let’s look at the distribution of categorical features. This helps us spot categories that may impact our target variable and identify any imbalances.\n\n**Why this matters:**\n- Reveals dominant groups (e.g., gender, work experience, degree type)\n- Helps identify potential bias or imbalance\n- Provides context for later analysis\n''')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        st.markdown(f"#### Distribution of {col}")
        fig = px.histogram(df, x=col, color=col, title=f'Distribution of {col}', template="plotly_white",
                          text_auto=True, width=600, height=350,
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(bargap=0.25, xaxis_title=col, yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=False)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>There are more male than female students in the program.</li>\n<li>Most students have <b>no prior work experience</b>.</li>\n<li>Comm&Mgmt is the most popular major, and most MBA students specialize in Mkt&Fin.</li>\n<li>There are more students who landed a job than those who did not.</li>\n<li>Imbalances in these categories may influence placement trends and model predictions.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*Now that we have a sense of the categorical landscape, let’s turn our attention to the academic and numerical achievements of these students. Understanding the spread and central tendency of these features will help us spot outliers and trends that could impact placement.*''')
    st.markdown('---')
    # 3. What are the distributions of numerical features?
    st.markdown('### 3. How are the numerical features such as academic percentages and entrance test scores distributed?')
    st.markdown('''Examining the spread of numerical features helps us spot outliers, detect extreme values, and understand variability and central tendency.\n\n**Why this matters:**\n- Outliers can affect modeling\n- Shows if features are normally distributed or skewed\n- Reveals the range and typical values for each metric\n''')
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        st.markdown(f"#### Distribution of {col}")
        fig = px.histogram(df, x=col, color='Placement_Status' if 'Placement_Status' in df.columns else None,
                          title=f'Distribution of {col}', template="plotly_white", width=600, height=350,
                          color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        fig.update_layout(bargap=0.25, xaxis_title=col, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=False)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>Most academic features are bell-shaped and nearly normal, with slight skewness.</li>\n<li><b>Salary</b> is highly right-skewed, with most values in the lower bracket and a few high outliers.</li>\n<li>Outlier analysis shows only a small percentage of outliers in most features, except for salary.</li>\n<li>Skewness analysis confirms most features are symmetric, except for salary.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*With a clear view of both categorical and numerical distributions, it’s important to understand how these features interact. Are there strong relationships between academic scores, or are they largely independent? The correlation matrix will help us answer this.*''')
    st.markdown('---')
    # 4. What are the linear relationships between the numerical features?
    st.markdown('### 4. What are the linear relationships between the numerical features?')
    st.markdown('''A correlation matrix shows the linear relationships between numerical variables.\n\n**Why this matters:**\n- Reveals which features move together\n- Identifies redundancy or independence\n- Helps in feature selection and engineering\n''')
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    correlation = df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, ax=ax, cbar_kws={'shrink':0.7})
    plt.title('Correlation Matrix')
    st.pyplot(fig)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>Academic percentages (secondary, higher secondary, degree, MBA) are moderately correlated, indicating students who perform well in one area tend to do well in others.</li>\n<li>There is little to no correlation between academic scores and salary, suggesting that high academic performance does not guarantee a higher salary offer.</li>\n<li>Low multicollinearity (VIF &lt; 5) means features are not redundant and can be used together in modeling.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*Having explored the relationships between features, let’s now focus on our main outcome of interest: the placement status. Understanding the balance between placed and not placed students is crucial for modeling and evaluation.*''')
    st.markdown('---')
    # 5. Is there a class imbalance in the target variable, Placement Status?
    st.markdown('### 5. Is there a class imbalance in the target variable, Placement Status?')
    st.markdown('''We examine the balance between students who were placed and those who were not.\n\n**Why this matters:**\n- Imbalanced classes can bias models\n- May require resampling (e.g., SMOTE)\n- Informs evaluation metrics\n''')
    fig, ax = plt.subplots(figsize=(5,3))
    sns.countplot(data=df, x='Placement_Status', palette=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_title('Distribution of Placement Status')
    ax.set_xlabel('Placement Status')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.write(df['Placement_Status'].value_counts(normalize=True))
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>The target variable is imbalanced: about 68% of students are placed, while 32% are not.</li>\n<li>This imbalance suggests the need for resampling techniques (like SMOTE) to ensure fair model training.</li>\n<li>Predictive models should be evaluated with metrics that account for imbalance (e.g., F1-score, ROC-AUC).</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*Now that we know the class balance, let’s dig deeper into how academic performance during school years relates to placement. Are there clear patterns or separations between those who are placed and those who are not?*''')
    st.markdown('---')
    # 6. How do academic performances during school years relate to job placement status?
    st.markdown('### 6. How do academic performances during school years relate to job placement status?')
    st.markdown('''Pairplots visually show the relationship between selected academic features and placement status, highlighting separation, clustering, and linearity.\n\n**Why this matters:**\n- Reveals if high academic scores lead to better placement\n- Shows overlap and exceptions\n- Helps identify strong predictors\n''')
    selected_cols = list(numerical_columns[:3]) + ['Placement_Status']
    pairplot_fig = sns.pairplot(df[selected_cols], hue='Placement_Status', hue_order=['Placed', 'Not Placed'],
                                palette={'Placed': '#1f77b4', 'Not Placed': '#ff7f0e'}, height=2.5)
    st.pyplot(pairplot_fig)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>Placed students (blue) generally have higher academic percentages, but there is overlap—some placed students have lower scores, and some not placed have higher scores.</li>\n<li>This suggests that while academic performance is important, it is not the sole determinant of placement.</li>\n<li>Other factors (like work experience, degree type, or soft skills) may also play a significant role.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*Academic performance is just one piece of the puzzle. Let’s now examine how the type of degree pursued by students influences their chances of being placed.*''')
    st.markdown('---')
    # 7. What are the placement rates for each type of degree?
    st.markdown('### 7. What are the placement rates for each type of degree?')
    st.markdown('''We analyze placement rates by degree type to see if academic background influences the likelihood of securing a job placement.\n\n**Why this matters:**\n- Identifies which degrees are most valued by employers\n- Helps students and advisors make informed decisions\n''')
    cat_col = 'Degree_Type' if 'Degree_Type' in df.columns else cat_cols[0]
    placement_rate = df.groupby(cat_col)['Placement_Status'].value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(7,4))
    placement_rate.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_title(f'Placement Rate by {cat_col}')
    ax.set_ylabel('Percentage')
    ax.set_xlabel(cat_col)
    ax.legend(title='Placement Status')
    st.pyplot(fig)
    st.dataframe(placement_rate, use_container_width=True, height=200)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>Students from <b>Comm&Mgmt</b> and <b>Sci&Tech</b> degrees have higher placement rates (&gt;70%).</li>\n<li>Students in the "Others" group have a lower placement rate (&lt;55%).</li>\n<li>This suggests that academic specialization is important for employability, and some degrees are more in demand by employers.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('''*Finally, let’s see how practical experience factors into placement. Does having work experience before the MBA really make a difference in employability?*''')
    st.markdown('---')
    # 8. Do having work experience affect a student's job placement?
    st.markdown('### 8. Do having work experience affect a student’s job placement?')
    st.markdown('''We explore how prior work experience impacts placement outcomes, providing actionable insights for students and university advisors.\n\n**Why this matters:**\n- Work experience may be a strong predictor of placement\n- Helps students understand the value of internships or jobs before MBA\n''')
    cat_col = 'Work_Experience' if 'Work_Experience' in df.columns else cat_cols[0]
    placement_rate_we = df.groupby(cat_col)['Placement_Status'].value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(7,4))
    placement_rate_we.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], ax=ax)
    ax.set_title(f'Placement Rate by {cat_col}')
    ax.set_ylabel('Percentage')
    ax.set_xlabel(cat_col)
    ax.legend(title='Placement Status')
    st.pyplot(fig)
    st.dataframe(placement_rate_we, use_container_width=True, height=200)
    st.markdown('''<div class="insight">\n<b>Insight:</b>\n<ul>\n<li>Students with prior work experience have a much higher placement rate (~86%) than those without (~59%).</li>\n<li>Work experience is a strong predictor of placement, but not a guarantee—some with experience are not placed, and some without are placed.</li>\n<li>This highlights the value of practical exposure and suggests that students should seek internships or jobs before their MBA to improve employability.</li>\n</ul>\n</div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('<div class="big-font section-title">Conclusion: What Have We Learned from the EDA?</div>', unsafe_allow_html=True)
    st.markdown('''
The exploratory data analysis has provided a comprehensive understanding of the factors influencing MBA student placements at the university. By examining both categorical and numerical features, their relationships, and the impact of academic and non-academic variables, we have built a strong foundation for predictive modeling. Key takeaways include:

- The dataset is well-structured, with clear academic, demographic, and placement information.
- There are imbalances in gender, work experience, and placement status that may affect modeling.
- Most academic features are nearly normal, but salary is highly skewed.
- Academic performance is important but not the sole determinant of placement; work experience and degree type also play significant roles.
- The target variable is imbalanced, requiring careful handling in modeling.
- No significant multicollinearity or problematic outliers were found, so all features can be used for modeling.
    ''')

    st.markdown('---')
    st.markdown('### 🔍 Having Explored the Data, Let\'s Briefly understand the model and make predictions for new students.')
    st.markdown('''
Before we move to the interactive prediction tool, let\'s recap the target variable and the modeling approach used in this project:

**Target Variable:**
- The target variable for our prediction is **Placement_Status**, which indicates whether a student is "Placed" (successfully secured a job) or "Not Placed" after graduation. This is a binary classification problem.

**Best Model Used:**
- After evaluating several machine learning algorithms—including Logistic Regression, Random Forest, Decision Tree, KNN, SVM, and XGBoost—the **XGBoost Classifier** was selected as the best model for this task.
- **Why XGBoost?**
    - XGBoost consistently outperformed other models in terms of accuracy, F1-score, and ROC-AUC, especially in handling the class imbalance present in the dataset.
    - It is robust to outliers, handles both numerical and categorical features well (after encoding), and provides feature importance for interpretability.
    - XGBoost\'s ability to handle missing values and its regularization features help prevent overfitting, making it ideal for this dataset.

With these choices, our model is well-equipped to predict placement outcomes for new students based on their academic, demographic, and experiential profiles.

---
                
#### Let's Predict Placement Outcomes for New Students!
                
Now, you can try the interactive prediction tool to estimate a student\'s placement outcome based on their profile`
''')

if __name__ == "__main__":
    run()