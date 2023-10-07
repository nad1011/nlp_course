import os

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


# Định nghĩa hàm load_imdb_data. Hàm này được sử dụng để đọc dữ liệu từ thư mục dataset. Nó duyệt qua tất cả các tệp
# trong thư mục ‘pos’ và ‘neg’, đọc nội dung của mỗi tệp (đánh giá) và gán nhãn tương ứng (‘pos’ hoặc ‘neg’).
def load_imdb_data(dir):
    labels = []
    reviews = []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(dir, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                review = file.read()
                reviews.append(review)
                labels.append(label)
    return pd.DataFrame({'review': reviews, 'label': labels})


def preprocessing_data(data):
    data['review'] = data['review'].str.replace('\W', ' ')
    data['review'] = data['review'].str.lower()
    stop_words = set(stopwords.words('english'))
    data['review'] = data['review'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    stemmer = PorterStemmer()
    data['review'] = data['review'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
    lemmatizer = WordNetLemmatizer()
    data['review'] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))
    return data


# Đọc dữ liệu từ thư mục dataset và lưu kết quả vào DataFrame df.
print("Reading data...")
data_dir = r'aclImdb\train'  # Đường dẫn tới thư mục dataset
df = load_imdb_data(data_dir)
test_data_dir = r'aclImdb\test'  # Đường dẫn đến thư mục test
test_df = load_imdb_data(test_data_dir)  # Sử dụng hàm load_imdb_data đã tạo trước đó
print("Done!")


# Tiến hành một số bước tiền xử lý trên dữ liệu, bao gồm loại bỏ các ký tự không phải chữ, chuyển đổi văn bản thành
# chữ thường, loại bỏ các từ không liên quan (stop words), stemming và lemmatization.
print("Start Preprocessing")
df = preprocessing_data(df)
test_df = preprocessing_data(test_df)
print("Done!")

print(df["review"])

x_train, y_train, x_test, y_test = df['review'], df['label'], test_df['review'], test_df['label']

# Sử dụng TfidfVectorizer để chuyển đổi văn bản thành vector số, sau đó huấn luyện một mô hình Naive Bayes
# (MultinomialNB) trên tập huấn luyện.
print("Vectorization and training")
# Biểu Diễn Dữ Liệu Văn Bản: Dữ liệu văn bản được biểu diễn thành các vectơ số học bằng cách sử dụng TfidfVectorizer.
# Cụ thể, TfidfVectorizer chuyển đổi các văn bản thành vectơ TF-IDF (Term Frequency-Inverse Document Frequency) để
# đại diện cho mức độ quan trọng của các từ trong văn bản.
tfidf_vectorizer = TfidfVectorizer(max_features=25000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

# Huấn Luyện Mô Hình: Bạn huấn luyện một mô hình phân loại Naive Bayes đa nominal (MultinomialNB) trên tập huấn
# luyện. Mô hình này sẽ học cách phân loại đánh giá là tích cực hoặc tiêu cực dựa trên dữ liệu huấn luyện.
model = LogisticRegression()
model.fit(x_train_tfidf, y_train)
print("Done!")

# Sử dụng mô hình đã được huấn luyện để dự đoán nhãn cho tập kiểm tra và
# in ra độ chính xác cũng như báo cáo phân loại.
y_pred = model.predict(x_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
# ****************************************************************************************************************

