import streamlit as st
st.set_page_config(
    page_title="Employee Layoff Analytics & Prediction Dashboard",
    page_icon="🤖",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

USER_FILE = "users.txt"

def load_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and ":" in line:
                    username, password = line.split(":", 1)
                    users[username] = password
    else:
        users["admin"] = "password"
        save_users(users)
    return users

def save_users(users):
    with open(USER_FILE, "w") as f:
        for username, password in users.items():
            f.write(f"{username}:{password}\n")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'users' not in st.session_state:
    st.session_state.users = load_users()

def login(username, password):
    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.authenticated = True
        st.session_state.current_user = username
        return True
    return False

def signup(username, password):
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = password
    save_users(st.session_state.users)
    return True

def logout():
    st.session_state.authenticated = False
    st.session_state.current_user = None

if not st.session_state.authenticated:
    st.title("🔐 Employee Layoff Analytics Dashboard - Login")
    
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
    
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if login(username, password):
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab_signup:
        with st.form("signup_form"):
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            confirm_pass = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif signup(new_user, new_pass):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists")
    
    st.stop()

col_title, col_logout = st.columns([6,1])
with col_title:
    st.title("🤖 Employee Layoff Analytics & Prediction Dashboard")
with col_logout:
    if st.button("Logout"):
        logout()
        st.rerun()

@st.cache_data
def generate_enhanced_sample_data():
    np.random.seed(42)
    n_rows = 3000
    
    companies = ['TechCorp', 'FinServe', 'HealthPlus', 'RetailGiant', 'AutoMakers']
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Automotive']
    cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai']
    roles = ['Software Engineer', 'Data Analyst', 'Product Manager', 'HR Manager', 'Sales Executive']
    
    df = pd.DataFrame({
        'Employee_ID': [f'EMP{str(i).zfill(5)}' for i in range(1, n_rows+1)],
        'Company': np.random.choice(companies, n_rows, p=[0.25, 0.2, 0.2, 0.2, 0.15]),
        'Industry': np.random.choice(industries, n_rows, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'City': np.random.choice(cities, n_rows),
        'Role': np.random.choice(roles, n_rows, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        'Age': np.random.normal(35, 8, n_rows).clip(22, 60).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_rows, p=[0.6, 0.4]),
        'Experience_Years': np.random.exponential(5, n_rows).clip(1, 25).astype(int),
        'Salary_LPA': np.random.lognormal(2.5, 0.4, n_rows).clip(4, 50),
        'Performance_Rating': np.random.choice(['Exceeds', 'Meets', 'Below'], n_rows, p=[0.2, 0.6, 0.2]),
        'Tenure_In_Company': np.random.exponential(3, n_rows).clip(1, 15).astype(int),
        'Projects_Completed': np.random.poisson(8, n_rows),
        'Promotions': np.random.binomial(3, 0.3, n_rows),
        'Layoff_Reason': np.random.choice(['Cost Cutting', 'Restructuring', 'Performance Issues', 
                                          'Company Closure', 'Automation'], n_rows),
        'Layoff_Year': np.random.choice([2021, 2022, 2023], n_rows, p=[0.3, 0.4, 0.3]),
    })
    
    df['Laid_Off'] = 0
    performance_weights = {'Exceeds': 0.05, 'Meets': 0.3, 'Below': 0.85}
    df['Performance_Weight'] = df['Performance_Rating'].map(performance_weights)
    industry_risk = {'Technology': 0.7, 'Finance': 0.4, 'Healthcare': 0.2, 'Retail': 0.8, 'Automotive': 0.5}
    df['Industry_Risk'] = df['Industry'].map(industry_risk)
    company_risk = {'TechCorp': 0.8, 'FinServe': 0.3, 'HealthPlus': 0.2, 'RetailGiant': 0.9, 'AutoMakers': 0.4}
    df['Company_Risk'] = df['Company'].map(company_risk)
    year_risk = {2021: 0.3, 2022: 0.7, 2023: 0.5}
    df['Year_Risk'] = df['Layoff_Year'].map(year_risk)
    
    df['Layoff_Probability'] = (
        0.35 * df['Performance_Weight'] +
        0.25 * df['Industry_Risk'] +
        0.15 * df['Company_Risk'] +
        0.10 * df['Year_Risk'] +
        0.05 * (df['Salary_LPA'] > df['Salary_LPA'].median()) +
        0.05 * (df['Experience_Years'] > 15) +
        0.05 * (df['Projects_Completed'] < 5) +
        np.random.normal(0, 0.05, n_rows)
    ).clip(0, 1)
    
    threshold = np.random.uniform(0.3, 0.5, n_rows)
    df['Laid_Off'] = (df['Layoff_Probability'] > threshold).astype(int)
    
    target_rate = 0.45
    current_rate = df['Laid_Off'].mean()
    if current_rate < target_rate - 0.05:
        n_needed = int((target_rate - current_rate) * len(df))
        candidates = df[df['Laid_Off'] == 0].nlargest(n_needed, 'Layoff_Probability')
        df.loc[candidates.index, 'Laid_Off'] = 1
    elif current_rate > target_rate + 0.05:
        n_remove = int((current_rate - target_rate) * len(df))
        candidates = df[df['Laid_Off'] == 1].nsmallest(n_remove, 'Layoff_Probability')
        df.loc[candidates.index, 'Laid_Off'] = 0
    
    df = df.drop(['Performance_Weight', 'Industry_Risk', 'Company_Risk', 
                  'Year_Risk', 'Layoff_Probability'], axis=1)
    return df

df = generate_enhanced_sample_data()

with st.sidebar:
    st.header("🔍 Filters")
    
    years = sorted(df['Layoff_Year'].unique())
    selected_years = st.multiselect("Select Years", options=years, default=years)
    
    reasons = df['Layoff_Reason'].unique()
    selected_reasons = st.multiselect("Select Layoff Reasons", options=reasons, default=reasons)
    
    industries = df['Industry'].unique()
    selected_industries = st.multiselect("Select Industries", options=industries, default=industries)
    
    st.subheader("Additional Filters")
    age_range = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()),
                          (int(df['Age'].min()), int(df['Age'].max())))
    experience_range = st.slider("Experience Range (Years)", int(df['Experience_Years'].min()), int(df['Experience_Years'].max()),
                                 (int(df['Experience_Years'].min()), int(df['Experience_Years'].max())))
    salary_range = st.slider("Salary Range (LPA)", float(df['Salary_LPA'].min()), float(df['Salary_LPA'].max()),
                             (float(df['Salary_LPA'].min()), float(df['Salary_LPA'].max())))
    
    st.markdown("---")
    st.header("➕ Manual Employee Entry")
    with st.form("manual_employee_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=22, max_value=60, value=35, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            experience = st.number_input("Experience (Years)", min_value=1, max_value=25, value=5, step=1)
            salary = st.number_input("Salary (LPA)", min_value=4.0, max_value=50.0, value=15.0, step=0.5)
        with col2:
            performance = st.selectbox("Performance Rating", ["Exceeds", "Meets", "Below"])
            industry = st.selectbox("Industry", sorted(df['Industry'].unique()))
            role = st.selectbox("Role", sorted(df['Role'].unique()))
            tenure = st.number_input("Tenure (Years)", min_value=1, max_value=15, value=3, step=1)
            projects = st.number_input("Projects Completed", min_value=0, max_value=30, value=8, step=1)
            promotions = st.number_input("Promotions", min_value=0, max_value=5, value=1, step=1)
        
        submitted = st.form_submit_button("View Employee Details", use_container_width=True)
        if submitted:
            st.session_state['manual_employee'] = {
                'Age': age,
                'Gender': gender,
                'Experience_Years': experience,
                'Salary_LPA': salary,
                'Performance_Rating': performance,
                'Industry': industry,
                'Role': role,
                'Tenure_In_Company': tenure,
                'Projects_Completed': projects,
                'Promotions': promotions
            }
            st.success("Employee details saved! Go to the '👤 Employee Analysis' tab.")

filtered_df = df[
    (df['Layoff_Year'].isin(selected_years)) &
    (df['Layoff_Reason'].isin(selected_reasons)) &
    (df['Industry'].isin(selected_industries)) &
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Experience_Years'] >= experience_range[0]) & (df['Experience_Years'] <= experience_range[1]) &
    (df['Salary_LPA'] >= salary_range[0]) & (df['Salary_LPA'] <= salary_range[1])
]

def feature_engineering(df, median_salary=None):
    df_engineered = df.copy()
    if median_salary is None:
        median_salary = df_engineered['Salary_LPA'].median()
    df_engineered['Salary_Experience_Ratio'] = df_engineered['Salary_LPA'] / (df_engineered['Experience_Years'] + 1)
    df_engineered['High_Performer'] = (df_engineered['Performance_Rating'] == 'Exceeds').astype(int)
    df_engineered['Low_Performer'] = (df_engineered['Performance_Rating'] == 'Below').astype(int)
    df_engineered['Productivity_Score'] = df_engineered['Projects_Completed'] / (df_engineered['Experience_Years'] + 1)
    df_engineered['Stagnant_Career'] = ((df_engineered['Experience_Years'] > 10) & (df_engineered['Promotions'] < 2)).astype(int)
    df_engineered['High_Cost_Employee'] = (df_engineered['Salary_LPA'] > median_salary).astype(int)
    df_engineered['Recent_Hire'] = (df_engineered['Tenure_In_Company'] < 2).astype(int)
    age_bins = [20, 30, 40, 50, 60]
    age_labels = ['20-30', '30-40', '40-50', '50-60']
    df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    return df_engineered, median_salary

def preprocess_data_for_ml(df, for_training=True):
    df_engineered, median_salary = feature_engineering(df)
    numeric_features = ['Age', 'Experience_Years', 'Salary_LPA', 'Tenure_In_Company',
                        'Projects_Completed', 'Promotions', 'Salary_Experience_Ratio', 'Productivity_Score']
    categorical_features = ['Gender', 'Performance_Rating', 'Industry', 'Role', 'Age_Group']
    binary_features = ['High_Performer', 'Low_Performer', 'Stagnant_Career', 'High_Cost_Employee', 'Recent_Hire']
    all_features = numeric_features + categorical_features + binary_features
    if for_training:
        X = df_engineered[all_features].copy()
        y = df_engineered['Laid_Off'].copy()
        return X, y, median_salary, all_features
    else:
        X = df_engineered[all_features].copy()
        return X, median_salary, all_features

def train_improved_model(X, y, model_type='XGBoost'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    if model_type == 'XGBoost' and XGB_AVAILABLE:
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              use_label_encoder=False, eval_metric='logloss')
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5,
                                       min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=5, subsample=0.8, random_state=42)
    elif model_type == 'Logistic Regression':
        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000,
                                   random_state=42, solver='liblinear')
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5,
                                       min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cv_mean_f1': cv_scores.mean(),
        'cv_std_f1': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'pipeline': pipeline,
        'model_type': model_type,
        'feature_names': list(X.columns)
    }
    
    try:
        if model_type in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                importance = pipeline.named_steps['classifier'].feature_importances_
                preprocessor = pipeline.named_steps['preprocessor']
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = cat_encoder.get_feature_names_out(categorical_features)
                all_features = list(numeric_features) + list(cat_features)
                if len(importance) == len(all_features):
                    metrics['feature_importance'] = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False).head(15)
    except:
        metrics['feature_importance'] = None
    
    return metrics

def predict_single_employee(pipeline, input_data, median_salary, feature_names):
    input_df = pd.DataFrame([input_data])
    input_df_engineered, _ = feature_engineering(input_df, median_salary)
    for feature in feature_names:
        if feature not in input_df_engineered.columns:
            if feature in ['High_Performer', 'Low_Performer', 'Stagnant_Career',
                           'High_Cost_Employee', 'Recent_Hire']:
                input_df_engineered[feature] = 0
            else:
                input_df_engineered[feature] = 0
    input_df_engineered = input_df_engineered[feature_names]
    try:
        prediction = pipeline.predict(input_df_engineered)[0]
        prediction_proba = pipeline.predict_proba(input_df_engineered)[0]
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0, [0.5, 0.5]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "🏢 Industry Analysis", "👥 Demographic Insights",
    "💰 Salary Analysis", "👤 Employee Analysis", "🤖 ML Prediction"
])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", f"{len(filtered_df):,}")
    with col2:
        laid_off = filtered_df['Laid_Off'].sum()
        st.metric("Employees Laid Off", f"{laid_off:,}")
    with col3:
        rate = (laid_off / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Layoff Rate", f"{rate:.1f}%")
    with col4:
        st.metric("Avg Salary (LPA)", f"₹{filtered_df['Salary_LPA'].mean():.1f}")
    
    chart_type = st.radio("Select chart type", ["Bar", "Pie"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Layoffs by Year")
        yearly = filtered_df.groupby('Layoff_Year')['Laid_Off'].sum().reset_index()
        if chart_type == "Bar":
            fig = px.bar(yearly, x='Layoff_Year', y='Laid_Off')
        else:
            fig = px.pie(yearly, values='Laid_Off', names='Layoff_Year')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Layoffs by Reason")
        reason_counts = filtered_df['Layoff_Reason'].value_counts().reset_index()
        reason_counts.columns = ['Layoff_Reason', 'Count']
        if chart_type == "Bar":
            fig = px.bar(reason_counts, x='Layoff_Reason', y='Count')
        else:
            fig = px.pie(reason_counts, values='Count', names='Layoff_Reason')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Layoffs by Industry")
        industry = filtered_df.groupby('Industry')['Laid_Off'].sum().reset_index().sort_values('Laid_Off', ascending=False)
        if chart_type == "Bar":
            fig = px.bar(industry, x='Industry', y='Laid_Off')
        else:
            fig = px.pie(industry, values='Laid_Off', names='Industry')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Top Companies by Layoffs")
        company = filtered_df.groupby('Company')['Laid_Off'].sum().reset_index().sort_values('Laid_Off', ascending=False).head(10)
        if chart_type == "Bar":
            fig = px.bar(company, x='Company', y='Laid_Off')
        else:
            fig = px.pie(company, values='Laid_Off', names='Company')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Industry-Wide Analysis")
    metric = st.selectbox("Sort by", ["Layoff_Rate", "Avg_Salary", "Total_Employees"])
    industry_stats = filtered_df.groupby('Industry').agg({
        'Laid_Off': ['sum', 'count', 'mean'],
        'Salary_LPA': 'mean',
        'Experience_Years': 'mean'
    }).round(2)
    industry_stats.columns = ['Laid_Off_Count', 'Total_Employees', 'Layoff_Rate',
                             'Avg_Salary', 'Avg_Experience']
    industry_stats = industry_stats.sort_values(metric, ascending=False)
    st.dataframe(industry_stats, use_container_width=True)
    fig = px.bar(industry_stats.reset_index(), x='Industry', y=metric,
                title=f'Industry Comparison - {metric}', color=metric)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Demographic Insights")
    col1, col2 = st.columns(2)
    with col1:
        gender_stats = filtered_df.groupby('Gender')['Laid_Off'].mean().reset_index()
        fig = px.bar(gender_stats, x='Gender', y='Laid_Off', title='Layoff Rate by Gender', color='Gender')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        norm = st.checkbox("Normalize histogram")
        fig = px.histogram(filtered_df, x='Age', color='Laid_Off',
                          title="Age Distribution by Layoff Status",
                          barmode='overlay', histnorm='percent' if norm else None)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Experience vs Salary by Layoff Status")
    fig = px.scatter(filtered_df, x='Experience_Years', y='Salary_LPA',
                     color='Laid_Off', symbol='Performance_Rating',
                     hover_data=['Role', 'Company'])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Salary Analysis")
    col1, col2 = st.columns(2)
    with col1:
        bins = st.slider("Number of bins", 10, 50, 30)
        fig = px.histogram(filtered_df, x='Salary_LPA', nbins=bins, title="Salary Distribution (LPA)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(filtered_df, x='Laid_Off', y='Salary_LPA', title="Salary by Layoff Status")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Average Salary by Industry")
    avg_salary = filtered_df.groupby('Industry')['Salary_LPA'].mean().reset_index().sort_values('Salary_LPA', ascending=False)
    fig = px.bar(avg_salary, x='Industry', y='Salary_LPA', color='Industry')
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("👤 Employee Analysis")
    
    if 'manual_employee' not in st.session_state:
        st.info("Please enter employee details in the sidebar form and click 'View Employee Details'.")
    else:
        emp = st.session_state['manual_employee']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Employee Information")
            st.markdown(f"""
            - **Age:** {emp['Age']}
            - **Gender:** {emp['Gender']}
            - **Experience:** {emp['Experience_Years']} years
            - **Salary:** ₹{emp['Salary_LPA']} LPA
            - **Performance Rating:** {emp['Performance_Rating']}
            """)
        with col2:
            st.markdown("### Job Details")
            st.markdown(f"""
            - **Industry:** {emp['Industry']}
            - **Role:** {emp['Role']}
            - **Tenure:** {emp['Tenure_In_Company']} years
            - **Projects Completed:** {emp['Projects_Completed']}
            - **Promotions:** {emp['Promotions']}
            """)
        
        st.markdown("---")
        st.markdown("### Comparison with Overall Workforce")
        
        comp_df = filtered_df
        metrics_to_compare = ['Age', 'Salary_LPA', 'Experience_Years', 'Tenure_In_Company', 'Projects_Completed', 'Promotions']
        comparison_data = []
        for metric in metrics_to_compare:
            emp_value = emp[metric]
            mean_val = comp_df[metric].mean()
            p25 = comp_df[metric].quantile(0.25)
            p50 = comp_df[metric].quantile(0.50)
            p75 = comp_df[metric].quantile(0.75)
            if emp_value < p25:
                standing = "Below 25th percentile"
            elif emp_value < p50:
                standing = "25th–50th percentile"
            elif emp_value < p75:
                standing = "50th–75th percentile"
            else:
                standing = "Above 75th percentile"
            comparison_data.append({
                'Metric': metric,
                'Employee Value': round(emp_value, 2),
                'Average': round(mean_val, 2),
                '25th %ile': round(p25, 2),
                '50th %ile': round(p50, 2),
                '75th %ile': round(p75, 2),
                'Standing': standing
            })
        
        comp_df_display = pd.DataFrame(comparison_data)
        st.dataframe(comp_df_display, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Layoff Risk Prediction")
        
        if 'model_trained' in st.session_state and st.session_state['model_trained']:
            pred, proba = predict_single_employee(
                st.session_state['metrics']['pipeline'],
                emp,
                st.session_state['median_salary'],
                st.session_state['feature_names']
            )
            risk = proba[1] * 100
            
            col1, col2 = st.columns([1, 2])
            with col1:
                if risk >= 70:
                    st.error(f"### High Risk\n**{risk:.1f}%**")
                elif risk >= 40:
                    st.warning(f"### Medium Risk\n**{risk:.1f}%**")
                else:
                    st.success(f"### Low Risk\n**{risk:.1f}%**")
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk,
                    title={'text': "Layoff Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk}
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Recommendations")
            if risk >= 70:
                st.write("""
                - **Immediate action required:** Schedule performance review.
                - Consider skill development or role adjustment.
                - Monitor closely for next quarter.
                """)
            elif risk >= 40:
                st.write("""
                - **Monitor performance** and provide coaching.
                - Encourage upskilling and cross-training.
                - Regular check-ins recommended.
                """)
            else:
                st.write("""
                - **Continue current trajectory.**
                - Seek growth opportunities within the company.
                - Maintain regular skill updates.
                """)
        else:
            st.info("No model trained yet. Please go to the 'ML Prediction' tab and train a model first.")

with tab6:
    st.subheader("Machine Learning Prediction Model")
    st.write("The model uses advanced features to predict layoff risk with high accuracy.")
    
    model_options = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
    if XGB_AVAILABLE:
        model_options.insert(0, 'XGBoost')
    model_type = st.selectbox("Select Model", model_options)
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            X, y, median_salary, feature_names = preprocess_data_for_ml(filtered_df)
            metrics = train_improved_model(X, y, model_type)
            st.session_state['metrics'] = metrics
            st.session_state['model_trained'] = True
            st.session_state['model_type'] = model_type
            st.session_state['median_salary'] = median_salary
            st.session_state['feature_names'] = feature_names
            st.success(f"✅ {model_type} model trained successfully!")
    
    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        metrics = st.session_state['metrics']
        st.subheader("Model Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        st.write(f"Cross-Validation F1 Score: {metrics['cv_mean_f1']:.3f} ± {metrics['cv_std_f1']:.3f}")
        
        st.subheader("Confusion Matrix")
        fig = px.imshow(metrics['confusion_matrix'], text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"), x=['Not Laid Off', 'Laid Off'], y=['Not Laid Off', 'Laid Off'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
        fig.update_layout(title=f'ROC Curve - {model_type}', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)
        
        if metrics.get('feature_importance') is not None:
            st.subheader("Top Important Features")
            fig = px.bar(metrics['feature_importance'], x='Importance', y='Feature', orientation='h', color='Importance')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Test on a Specific Employee from Dataset")

        employee_ids = df['Employee_ID'].tolist()
        selected_employee_id = st.selectbox("Choose an Employee ID", employee_ids)

        if selected_employee_id:
            employee_row = df[df['Employee_ID'] == selected_employee_id].iloc[0]

            emp_from_dataset = {
                'Age': employee_row['Age'],
                'Gender': employee_row['Gender'],
                'Experience_Years': employee_row['Experience_Years'],
                'Salary_LPA': employee_row['Salary_LPA'],
                'Performance_Rating': employee_row['Performance_Rating'],
                'Industry': employee_row['Industry'],
                'Role': employee_row['Role'],
                'Tenure_In_Company': employee_row['Tenure_In_Company'],
                'Projects_Completed': employee_row['Projects_Completed'],
                'Promotions': employee_row['Promotions']
            }

            actual_status = employee_row['Laid_Off']
            st.write(f"**Actual Layoff Status in Dataset:** {'Laid Off' if actual_status == 1 else 'Not Laid Off'}")

            pred, proba = predict_single_employee(
                st.session_state['metrics']['pipeline'],
                emp_from_dataset,
                st.session_state['median_salary'],
                st.session_state['feature_names']
            )
            risk = proba[1] * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", "Laid Off" if pred == 1 else "Not Laid Off")
                st.metric("Predicted Probability of Layoff", f"{risk:.2f}%")
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk,
                    title={'text': "Layoff Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk}
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if pred == actual_status:
                st.success("✅ Prediction matches actual status.")
            else:
                st.warning("⚠️ Prediction does not match actual status. (This can happen due to model limitations or data noise.)")
    else:
        st.info("👈 Please train a model first to enable predictions on specific employees.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <h4>👥 Team Members</h4>
        <p>
            <strong>G. NARENDRA</strong> - Data Scientist<br>
            <strong>D. MYTHRI</strong> - ML Engineer<br>
            <strong>P. JOHN NIHAS</strong> - Frontend Developer<br>
        </p>
        <p>© 2026 Employee Layoff Analytics Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not XGB_AVAILABLE:
    st.sidebar.info("💡 Install XGBoost for better accuracy: `pip install xgboost`")