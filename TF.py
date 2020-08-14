###########################
# 라이브러리 사용
import tensorflow as tf
import pandas as pd

###########################
# 데이터를 준비합니다.
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
레모네이드 = pd.read_csv(파일경로)
레모네이드.head()
# 종속변수, 독립변수
독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)

###########################
# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1])  # [독립변수의 column의 개수]
Y = tf.keras.layers.Dense(1)(X)  # (종속변수의 column의 개수)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

###########################
# 모델을 학습시킵니다. 
# model.fit(독립, 종속, epochs=1000, verbose=0) # verbose = 0 == 출력없이 학습
model.fit(독립, 종속, epochs=10)  # epochs = 전체 데이터를 몇 번 반복해서 학습할 것인가?

# Epoch 1/10 // 현재 횟수
# 1/1 [==============================] - 0s 1ms/step - loss: 3382.7236 // 시간 // 학습진행률
# Epoch 2/10
# 1/1 [==============================] - 0s 4ms/step - loss: 3379.9939
# Epoch 3/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3377.2646
# Epoch 4/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3374.5364
# Epoch 5/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3371.8098
# Epoch 6/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3369.0842
# Epoch 7/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3366.3594
# Epoch 8/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3363.6357
# Epoch 9/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3360.9128
# Epoch 10/10
# 1/1 [==============================] - 0s 2ms/step - loss: 3358.1917

# Error == (종속변수 - 예측값)
# loss = Error^2 / len(Error)

###########################
# 모델을 이용합니다. 
print(model.predict(독립))
print(model.predict([[15]]))
