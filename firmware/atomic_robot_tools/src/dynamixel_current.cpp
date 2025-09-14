#include <Arduino.h>
#include <Dynamixel2Arduino.h>
#include <M5AtomS3.h>


// Dynamixel2Arduino ////////////////////////////////////////////////////////
#define DXL_SERIAL Serial1
#define SERIAL_CONFIG SERIAL_8N1

using namespace ControlTableItem;

const long BAUDRATE = 2000000; // 2Mが限界だった
const int TIMEOUT = 100;  //ms
const uint8_t DXL_ID = 0;
const float DXL_PROTOCOL_VERSION = 2.0;

const int EN_PIN = 6;  // Enable Pin
const int RX_PIN = 5;
const int TX_PIN = 38;

Dynamixel2Arduino dxl(DXL_SERIAL, EN_PIN);

// motorStateは、0: 初期ストップ, 1: 正転中, 2: 正転後のストップ, 3: 逆転中
int motorState = 0;
String message;
unsigned long lastDisplayTime = 0;
const unsigned long displayInterval = 500; // [ms]
bool isAutoMode = false; // trueだと、autoModeInterval[ms]おきにmotorStateが切り替わる。falseだとクリック時に切り替わる。
const unsigned long autoModeInterval = 3000; // [ms]

void setupDXL()
{
  DXL_SERIAL.begin(BAUDRATE, SERIAL_CONFIG, RX_PIN, TX_PIN, false, TIMEOUT);
  dxl.begin(BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  dxl.torqueOff(DXL_ID);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT);
  dxl.torqueOn(DXL_ID);

  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 0);
}
/////////////////////////////////////////////////////////////////////////////

float readVoltage() {
  int32_t voltage_raw = dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, DXL_ID);
  return (float)voltage_raw / 10.0; // Unit [V]
}

float readTemperature() {
  int32_t temperature = dxl.readControlTableItem(PRESENT_TEMPERATURE, DXL_ID);
  return (float)temperature; // Unit [C]
}

void displayDXL(unsigned long interval) {
  unsigned long currentTime = millis();
  if (currentTime - lastDisplayTime >= interval) {
    lastDisplayTime = currentTime;
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.println(message);
    M5.Lcd.printf("\n%.2f [V]\n", readVoltage());
    M5.Lcd.printf("%d [mA]\n", (int)dxl.getPresentCurrent(DXL_ID, UNIT_MILLI_AMPERE));
    M5.Lcd.printf("%d [C]\n", (int)readTemperature());
  }
}

void setup() {
  M5.begin();
  M5.Lcd.setRotation(0);  // 画面向き設定（USB位置基準 0：下/ 1：右/ 2：上/ 3：左）
  M5.Lcd.setTextSize(2);  // 文字サイズ（整数倍率）

  setupDXL();

  message = "Stop\n\nPress to\nForward";
  displayDXL(0);
}

void loop() {
  if (!isAutoMode) {
    M5.update();
  }
  if (isAutoMode || M5.BtnA.wasReleased()) {
    switch (motorState) {
      case 0: // 「初期ストップ」の状態でボタンが押されると、正転を開始
        dxl.setGoalCurrent(DXL_ID, 500, UNIT_MILLI_AMPERE);
        message = "Forward\n\nPress to\nStop";
        motorState = 1;
        break;
      case 1: // 「正転中」の状態でボタンが押されると、モーターをストップ
        dxl.setGoalCurrent(DXL_ID, 0, UNIT_MILLI_AMPERE);
        message = "Stop\n\nPress to\nReverse";
        motorState = 2;
        break;
      case 2: // 「正転後のストップ」の状態でボタンが押されると、逆転を開始
        dxl.setGoalCurrent(DXL_ID, -800, UNIT_MILLI_AMPERE);
        message = "Reverse\n\nPress to\nStop";
        motorState = 3;
        break;
      case 3: // 「逆転中」の状態でボタンが押されると、モーターをストップさせ、初期状態に戻る
        dxl.setGoalCurrent(DXL_ID, 0, UNIT_MILLI_AMPERE);
        message = "Stop\n\nPress to\nForward";
        motorState = 0;
        break;
    }
  }

  // displayInterval[ms]おきに描画
  if (isAutoMode) {
    unsigned long startTime = millis();
    while (millis() - startTime < autoModeInterval) {
      displayDXL(displayInterval);
      delay(10);
    }
  } else {
    delay(10);
    displayDXL(displayInterval);
  }
}
