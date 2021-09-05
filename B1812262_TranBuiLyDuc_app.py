from B1812262_TranBuiLyDuc_train import classify
from flask import Flask, request, render_template

# Khởi tạo server backend
app = Flask(__name__, template_folder='templates')

# Đường dẫn chạy mô hình phân lớp
@app.route('/prediction', methods=['GET'])
def prediction():
    try:
        # Lấy dữ liệu người dụng nhập vào qua phương thức GET
        sepel_length = request.args.get('sepel-length')
        sepel_width = request.args.get('sepel-width')
        petal_length = request.args.get('petal-length')
        petal_width = request.args.get('petal-width')

        # Sử dụng dữ liệu để phân lớp bằng mô hình đã huấn luyện
        result = classify(sepel_length, sepel_width, petal_length, petal_width)
        return render_template('prediction.html', result = result)
    except:
        return 'Lỗi! Vui lòng nhập đầy đủ số liệu...'


# Trang chủ dẩn đến home
@app.route('/')
def home():
    return render_template('home.html') # Render home.html

# Start server
if __name__ == '__main__':
    app.run(debug=True)
