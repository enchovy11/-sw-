// 모터 핀 정의
const int leftMotorPin1 = 3;
const int leftMotorPin2 = 4;
const int rightMotorPin1 = 6;
const int rightMotorPin2 = 5;
const int steeringMotorPin1 = 7;
const int steeringMotorPin2 = 8;
// 가변저항 핀 정의
const int potentiometerPin = A0;
// 속도 제어 변수
int motorSpeed = 255; // 0-255 사이의 값
int steeringSpeed = 100; // 조향 모터 속도
// 조향 각도 배열
int steeringValues[7] = {0, 0, 0, 0, 0, 0, 0};
const int centerIndex = 3; // 중앙 위치의 인덱스
int currentSteeringIndex = centerIndex;
int leftMax = 0; // 좌측 최대값
int rightMin = 1023; // 우측 최소값
int centerVal; // 중앙값

void setup() {
  Serial.begin(115200);
  
  // 모터 핀 설정
  pinMode(leftMotorPin1, OUTPUT);
  pinMode(leftMotorPin2, OUTPUT);
  pinMode(rightMotorPin1, OUTPUT);
  pinMode(rightMotorPin2, OUTPUT);
  pinMode(steeringMotorPin1, OUTPUT);
  pinMode(steeringMotorPin2, OUTPUT);
  
  calibrateSteering();
  
  moveSteeringMotor(steeringValues[centerIndex]);
}

void loop() {
  if (Serial.available() > 0) {
    int input = Serial.read() - '0';
    if (input >= 1 && input <= 7) {
      currentSteeringIndex = input - 1;
      moveSteeringMotor(steeringValues[currentSteeringIndex]);
      
    }
    if (input == 9){
      moveForward(); // 지속적으로 전진
    }
    if (input == 10){
      stop();
    }
  }
  delay(10);
}

void calibrateSteering() {
  // 왼쪽으로 최대한 이동
  analogWrite(steeringMotorPin1, 255);
  analogWrite(steeringMotorPin2, 0);
  delay(2000); // 2초 동안 이동
  
  // 왼쪽 최대값 측정 (여러 번 측정하여 평균 계산)
  leftMax = 0;
  for (int i = 0; i < 10; i++) {
    leftMax += analogRead(potentiometerPin);
    delay(50);
  }
  leftMax /= 10;
  steeringValues[0] = leftMax;
  
  // 오른쪽으로 최대한 이동
  analogWrite(steeringMotorPin1, 0);
  analogWrite(steeringMotorPin2, 255);
  delay(4000); // 4초 동안 이동 (왼쪽에서 오른쪽으로)
  
  // 오른쪽 최소값 측정 (여러 번 측정하여 평균 계산)
  rightMin = 1023;
  for (int i = 0; i < 10; i++) {
    int value = analogRead(potentiometerPin);
    if (value < rightMin) rightMin = value;
    delay(50);
  }
  steeringValues[4] = rightMin;
  
  // 중앙값 계산
  centerVal = (leftMax + rightMin) / 2;
  steeringValues[centerIndex] = centerVal;
  
  // 모터 정지
  digitalWrite(steeringMotorPin1, LOW);
  digitalWrite(steeringMotorPin2, LOW);
  
  // 나머지 값들 계산
  for (int i = 1; i < 3; i++) {
    steeringValues[i] = map(i, 0, 3, leftMax, centerVal);
  }
  for (int i = 4; i < 6; i++) {
    steeringValues[i] = map(i, 3, 6, centerVal, rightMin);
  }
  
}

void moveForward() {
  analogWrite(leftMotorPin1, motorSpeed);
  analogWrite(leftMotorPin2, 0);
  analogWrite(rightMotorPin1, motorSpeed);
  analogWrite(rightMotorPin2, 0);
}

void moveSteeringMotor(int targetValue) {
  int currentValue = analogRead(potentiometerPin);
  while (abs(currentValue - targetValue) > 10) { // 10은 허용 오차
    if (currentValue < targetValue) {
      // 우회전
      analogWrite(steeringMotorPin1, steeringSpeed);
      analogWrite(steeringMotorPin2, 0);
    } else {
      // 좌회전
      analogWrite(steeringMotorPin1, 0);
      analogWrite(steeringMotorPin2, steeringSpeed);
    }
    currentValue = analogRead(potentiometerPin);
    delay(10);
  }
  // 모터 정지
  analogWrite(steeringMotorPin1, 0);
  analogWrite(steeringMotorPin2, 0);
}

void stop(){
  analogWrite(leftMotorPin1, 0);
  analogWrite(leftMotorPin2, 0);
  analogWrite(rightMotorPin1, 0);
  analogWrite(rightMotorPin2, 0);
  analogWrite(steeringMotorPin1, 0);
  analogWrite(steeringMotorPin2, 0);
}
