import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def preprocess_data(data):
    if isinstance(data, pd.DataFrame):
        preprocessed_data = [post.split() for post in data['text']]
        return preprocessed_data
    else:
        raise TypeError("Data must be a pandas DataFrame.")

def analyze_threats(data):
    if data:
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform([' '.join(post) for post in data])
        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        cluster_centers = kmeans.cluster_centers_
        terms = vectorizer.get_feature_names()
        top_terms = []
        for i in range(num_clusters):
            top_terms.append([terms[ind] for ind in cluster_centers[i].argsort()[-5:]])
        return top_terms
    else:
        raise ValueError("Data is empty.")

def recommend_security_measures(top_terms):
    if top_terms:
        recommendations = []
        for i, terms in enumerate(top_terms):
            recommendations.append(f"Recommendation for threat cluster {i+1}: Monitor network traffic for keywords {' '.join(terms)}")
        return recommendations
    else:
        return ["No threats identified."]

def main():
    try:
        # Load cybersecurity forum data
        data = pd.read_csv("cyberbullying_data.csv")
        
        # Preprocess the data
        preprocessed_data = preprocess_data(data)
        
        # Analyze threats
        top_terms = analyze_threats(preprocessed_data)
        
        # Recommend security measures
        recommendations = recommend_security_measures(top_terms)
        
        # Print recommendations
        print("Security Recommendations:")
        for recommendation in recommendations:
            print(recommendation)
    except FileNotFoundError:
        print("Error: Data file not found.")
    except pd.errors.EmptyDataError:
        print("Error: Data file is empty.")
    except pd.errors.ParserError:
        print("Error: Invalid file format. Unable to parse the data file.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
