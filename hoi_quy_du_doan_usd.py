import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns

# Đọc dữ liệu từ file CSV
# data = pd.read_csv('C:/Nam4/HocMay/VND=USD.csv')
data = pd.read_csv(
    'D:/Tài liệu/Kì 7/Học máy/BTL_Học Máy_CNTT1401_Dự Đoán Giá Ngoại Tệ Hồi Quy/VND=USD.csv')

data.info()

# sns.pairplot(data)
# sns.histplot(data['usd_price'])

# Chuyển đổi cột 'date' thành định dạng ngày tháng
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Tạo cột 'day_of_year' từ chỉ số ngày trong năm
data['day_of_year'] = data['date'].dt.dayofyear

# sns.heatmap(data.corr(), annot=True)
# plt.show()

# # Chọn biến phụ thuộc và biến độc lập
X = data[['day_of_year', 'interest_rate', 'cpi', 'pmi', 'unemployment']]
y = data['usd_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)

# # Tạo mô hình hồi quy tuyến tính đa biến
model = LinearRegression()
model.fit(X_train, y_train)

# # Dự đoán giá ngoại tệ từ dữ liệu đào tạo
train_predictions = model.predict(X)

# Dự đoán giá ngoại tệ mới dựa trên ngày và các biến độc lập
# Thay ngày tương ứng
new_date = datetime.strptime('9/6/2023', '%m/%d/%Y')
new_day_of_year = new_date.timetuple().tm_yday
# Thay các giá trị tương ứng
new_data = [[new_day_of_year, 5.5, 3.7, 48, 3.8]]
predicted_price = model.predict(new_data)

new_date2 = datetime.strptime('9/7/2023', '%m/%d/%Y')
new_day_of_year2 = new_date2.timetuple().tm_yday
new_data2 = [[new_day_of_year2, 5.5, 3.7, 48, 3.8]]
predicted_price2 = model.predict(new_data2)

new_date3 = datetime.strptime('9/8/2023', '%m/%d/%Y')
new_day_of_year3 = new_date3.timetuple().tm_yday
new_data3 = [[new_day_of_year3, 5.5, 3.7, 48, 3.8]]
predicted_price3 = model.predict(new_data3)

print('Giá ngoại tệ dự đoán ngày 6/9/2023:', predicted_price)
print('Giá ngoại tệ dự đoán ngày 7/9/2023:', predicted_price2)
print('Giá ngoại tệ dự đoán ngày 8/9/2023:', predicted_price3)

plt.figure(figsize=(15, 15))
# Tạo biểu đồ trực quan cho giá ngoại tệ dự đoán và giá ngoại tệ thực tế
plt.plot(data['date'], data['usd_price'],
         label='Actual foreign currency prices')  # giá ngoại tệ thực tế
plt.plot(data['date'], train_predictions,
         label='Predicted foreign currency prices (training)')  # Giá ngoại tệ dự đoán (đào tạo)
plt.axvline(x=new_date, color='r', linestyle='--',
            label='Predicted date')  # Ngày dự đoán
plt.xlabel('Day')
plt.ylabel('Foreign currency prices')
plt.title('Forecasting foreign currency prices')
plt.legend()
plt.show()
