from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model


app = Flask(__name__)

# 1. 개별 모델 및 스케일러 불러오기

# 3일 뒤 예측 모델
model1 = load_model("./int_model/lstm_3days_model.h5")
scaler_X3 = joblib.load("./int_model/scaler_X3.pkl")
scaler_y3 = joblib.load("./int_model/scaler_y3.pkl")

# 5일 뒤 예측 모델
model2 = load_model("./int_model/lstm_5days_model.h5")
scaler_X5 = joblib.load("./int_model/scaler_X5.pkl")
scaler_y5 = joblib.load("./int_model/scaler_y5.pkl")

# 공정A
model3 = load_model("./int_model/model-temp-press1.h5")
scalerA = joblib.load('./int_model/scaler1.pkl')

# 공정B
model4 =load_model("./int_model/model-high_temp31.h5")
scalerB = joblib.load('./int_model/scaler2.pkl')




# 2. 유저 입력 ->  모델에 맞는 입력형태로 만드는 변환 메서드들 개별 지정

# 2-1. 원자재 발주량 예측 모델(3,5일 뒤) 입력값 변환
def convert_user_input_to_lstm_input(orders):
    df = pd.DataFrame(orders)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    if 'qty' not in df.columns:
        raise ValueError("'qty' 필드가 없습니다.")

    if len(df) != 3:
        raise ValueError("정확히 3일치 데이터를 입력해주세요.")

    x3_values = df['qty'].values.reshape(1, -1) 
    x3_scaled = scaler_X3.transform(x3_values).reshape(-1, 3, 1) 

    x5_values = df['qty'].values.reshape(1, -1) 
    x5_scaled = scaler_X5.transform(x5_values).reshape(-1, 3, 1) 

    return x3_scaled, x5_scaled


# 2-2. 온도 / 압력 기반 모델 입력값 변환
def convert_user_input_to_model2_input(orders):

    # 기초 데이터프레임화
    df = pd.DataFrame(orders)

    # 필요한 입력값 정의
    temp = df['temp']
    press = df['press']
    mfm = temp**2+press**3
    mtm = temp**3-press**2
    
    features = []

    features.append([
    temp,
    press,
    temp*press,
    mfm,
    mtm
    ])
    
    # 데이터 정규화
    input = np.array(features).reshape(1,5)
    input_data = pd.DataFrame(input, 
                              columns=['Temperature (°C)', 'Pressure (kPa)', 
                                       'Temperature x Pressure', 'Material Fusion Metric',
                                       'Material Transformation Metric'])

    scaled_input = scalerA.transform(input_data)

    return scaled_input


# 2-3. 상열온도 세부 항목 기반 모델 입력값 변환
def convert_user_input_to_model3_input(orders):

    # 기초 데이터프레임화
    df = pd.DataFrame(orders)

    # 필요한 입력값 정의
    lTemp = df['leftHighTemp'].values[0]
    mTemp = df['midHighTemp'].values[0]
    msTemp = mTemp + 1.79086
    rTemp = df['rightHighTemp'].values[0]
    
    features = []

    features.append([
    lTemp,
    msTemp,
    mTemp,
    rTemp,
    mTemp-lTemp,
    mTemp-rTemp,
    np.mean([lTemp,mTemp,msTemp,rTemp])
    ])
    
    # 데이터 정규화
    input = np.array(features).reshape(1,7)
    input_data = pd.DataFrame(input, 
                              columns=['상열온도_좌_종료', '상열온도_중_시작', 
                                       '상열온도_중_종료', '상열온도_우_종료',
                                       '상열온도_좌중차','상열온도_우중차',
                                       '상열온도_평균'])
    scaled_input = scalerB.transform(input_data)

    return scaled_input


# 3. 예측 API 엔드포인트(각 모델별로 필요)

# 3-1. 발주량 예측 모델 사용
@app.route('/predictD3', methods=['POST'])
def predict_d3():
    try:
        data = request.get_json()
        orders = data.get('orders') 

        if not orders:
            return jsonify({'error': 'No orders provided'}), 400

        model_input, noUse_input = convert_user_input_to_lstm_input(orders)
        prediction = model1.predict(model_input)

        # 역정규화
        y3_pred = scaler_y3.inverse_transform(prediction).flatten()[0]

        return jsonify({'prediction': float(y3_pred)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predictD5', methods=['POST'])
def predict_d5():
    try:
        data = request.get_json()
        orders = data.get('orders') 

        if not orders:
            return jsonify({'error': 'No orders provided'}), 400

        noUse_input, model_input = convert_user_input_to_lstm_input(orders)
        prediction = model2.predict(model_input)

        # 역정규화
        y5_pred = scaler_y5.inverse_transform(prediction).flatten()[0]

        return jsonify({'prediction': float(y5_pred)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 3-2. 온도/압력 기반 모델 사용
@app.route('/predictA', methods=['POST'])
def predictA():
    try:
        data = request.get_json()
        # 클라이언트가 보낸 JSON데이터를 딕셔너리 형태로 파싱.
        # {"orders" : [{"date" : "2025-04-09", "qty" : 100}, ...]}

        orders = data.get('orders')
        # JSON에서 orders의 키에 해당하는 값(리스트 형태)로 가져옴.

        # 입력 검증
        if not orders:
            return jsonify({'error': 'No orders provides'}), 400
        
        model_input = convert_user_input_to_model2_input(orders)
        y_pred = model3.predict(model_input).flatten()[0] # 모델 입력값으로 예측 개시

        return jsonify({'prediction' : float(y_pred)}) # 여측결과를 JSON으로 감싸서 반환
    
    except Exception as e :
        return jsonify({'error': str(e)}), 500
    # 오류 발생시 500에러로 에러 메시지와 함께 반환.


    # 3-3. 상열온도 세부 항목 기반 모델 사용
@app.route('/predictB', methods=['POST'])
def predictB():
    try:
        data = request.get_json()
        # 클라이언트가 보낸 JSON데이터를 딕셔너리 형태로 파싱.
        # {"orders" : [{"date" : "2025-04-09", "qty" : 100}, ...]}

        orders = data.get('orders')
        # JSON에서 orders의 키에 해당하는 값(리스트 형태)로 가져옴.

        # 입력 검증
        if not orders:
            return jsonify({'error': 'No orders provides'}), 400
        
        model_input = convert_user_input_to_model3_input(orders)
        y_pred = model4.predict(model_input).flatten()[0] # 모델 입력값으로 예측 개시

        return jsonify({'prediction' : float(y_pred)}) # 여측결과를 JSON으로 감싸서 반환
    
    except Exception as e :
        return jsonify({'error': str(e)}), 500
    # 오류 발생시 500에러로 에러 메시지와 함께 반환.


# 4. 서버 실행
if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
# 기본적으로 localhost:5000에서 실행(port값 생략가능)
# 기본값은 host='127.0.0.1' -> 오직 본인 컴퓨터(로컬)에서만 접속 가능.
# host='0.0.0.0'이면 같은 네트워크내 다른 기기에서도 접속 가능.
# 외부에서 접속 가능하게 하려면 host='0.0.0.0' 반드시 사용.
# debug=True로 상시 업데이트 적용.
