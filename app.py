import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the data
data = pd.read_csv('tkeq_data_mth.csv')
metrics = ['date', 'spend', 'purchases']
data = data[metrics]

# model
degree = 2
X = data[['spend']]
y = data['purchases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
model = LinearRegression()
model.fit(X_poly_train, y_train)
y_pred = model.predict(X_poly_test)
mae = np.floor(mean_absolute_error(y_test, y_pred))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    spend_value = float(request.json['spend_value'])
    spend_value_poly = poly.transform([[spend_value]])
    predicted_purchases = int(np.ceil(model.predict(spend_value_poly)[0]))
    cpa1 = spend_value / (predicted_purchases + mae)
    cpa2 = spend_value / (predicted_purchases - mae)
    
    # Visualization
    plt.figure(figsize=(6, 3.8))
    sns.scatterplot(data=data, x='spend', y='purchases', label='Actual Data', color='blue')
    plt.scatter(spend_value, predicted_purchases, color='red', s=100, label=f'Predicted for Spend=${spend_value:,.0f}')
    spend_range = np.linspace(data['spend'].min(), data['spend'].max(), 500).reshape(-1, 1)
    plt.title("Spend vs Purchases with Prediction")
    plt.xlabel("Spend ($)")
    plt.ylabel("Purchases")
    plt.legend(loc='upper left')
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    # Result text
    result_text = (f'{predicted_purchases} ± {mae:.0f} purchases and a CPA between '
                   f'${cpa1:.1f} - ${cpa2:.1f} are forecast for a spend of ${spend_value:,.0f}')
    
    return jsonify(result_text=result_text, plot_url=plot_url)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
