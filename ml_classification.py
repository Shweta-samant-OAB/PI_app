import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def classify_products(df):
    """
    Classify products in Men, Unknown and Shoe categories using ML
    """
    # Filter products that need classification
    to_classify = df[df['Category'].isin(['Men', 'Unknown', 'Shoe'])]
    
    # Create features from product descriptions
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(to_classify['Product Story'].fillna(''))
    
    # Create target variable based on existing categories
    y = to_classify['Category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Predict on all unclassified products
    predictions = clf.predict(X)
    
    # Update the dataframe with new classifications
    df.loc[to_classify.index, 'Category'] = predictions
    
    return df

def get_feature_importance(df):
    """
    Get feature importance from the classifier
    """
    # Filter products that need classification
    to_classify = df[df['Category'].isin(['Men', 'Unknown', 'Shoe'])]
    
    # Create features from product descriptions
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(to_classify['Product Story'].fillna(''))
    
    # Create target variable
    y = to_classify['Category']
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Get feature importance
    feature_names = vectorizer.get_feature_names_out()
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance.head(20)  # Return top 20 most important features 