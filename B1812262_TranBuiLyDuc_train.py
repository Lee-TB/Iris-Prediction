# Nạp các gói thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

# Đọc tập tin json chứa tập dữ liệu iris
iris = pd.read_csv('./iris.csv')
X = iris.drop(columns=['variety'])
y = iris.variety

# Sử dụng nghi thức kiểm tra hold-out
# Chia dữ liệu ngẫu nhiên thành 2 tập dữ liệu con:
# training set và test set theo tỷ lệ 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Xây dựng mô hình với giải thuật Cây quyết định
model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)

# Hàm phân lớp dựa vào dữ liệu thuộc tính nhập vào
def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = model.predict(query)[0] # Retrieve from dictionary ex: ['setosa'] to 'setosa'
    return prediction # Return the prediction
