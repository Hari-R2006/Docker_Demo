from flask import Flask, request, render_template_string
import pickle
import numpy as np
app = Flask(__name__)
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('svr_model.pkl', 'rb') as f:
    svr_model = pickle.load(f)
#HTML form
form_html="""
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Prediction</title>
</head>
<body>
    <Center>
    <h1>Student Performance Prediction</h1>
    <form action="/predict" method="POST">
    <label for="hours">Hours Studied:</label>
    <input type="number" id="hours" name="hours" required><br><br>
    <input type="submit" value="Predict">
    <br><br>
    <div id="result"></div>
    </form>
    </Center>
</body>
</html>
"""
@app.route('/')
def index():
    return form_html
@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form.get('hours'))
        input_data = np.array([[hours]])
        lr_pred = lr_model.predict(input_data)[0]
        svr_pred = svr_model.predict(input_data)[0]

        return render_template_string(
            f"""
            <center>
            <p>Linear Regression Prediction: {lr_pred:.2f}</p>
            <p>SVR Prediction: {svr_pred:.2f}</p>
            </center>
            <a href="/">Go Back</a>
            """,
            lr_pred=lr_pred,
            svr_pred=svr_pred
        )
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    # run on port 9090 
    app.run(host='0.0.0.0', port=9090)
