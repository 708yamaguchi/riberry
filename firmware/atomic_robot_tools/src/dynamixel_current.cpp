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

// モーターの状態を管理するための変数を定義
// 0: 初期ストップ
// 1: 正転中
// 2: 正転後のストップ
// 3: 逆転中
int motorState = 0;

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


void setup() {
  M5.begin();
  M5.Lcd.setRotation(0);  // 画面向き設定（USB位置基準 0：下/ 1：右/ 2：上/ 3：左）
  M5.Lcd.setTextSize(2);  // 文字サイズ（整数倍率）

  setupDXL();

  M5.Lcd.clear();
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.print("Stop\n\nPress to\nForward");
}


void loop() {
  M5.update();
  if (M5.BtnA.wasReleased()) {
    switch (motorState) {
      case 0: // 「初期ストップ」の状態でボタンが押されると、正転を開始
        dxl.setGoalCurrent(DXL_ID, 500, UNIT_MILLI_AMPERE);
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.print("Forward\n\nPress to\nStop");
        motorState = 1;
        break;
      case 1: // 「正転中」の状態でボタンが押されると、モーターをストップ
        dxl.setGoalCurrent(DXL_ID, 0, UNIT_MILLI_AMPERE);
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.print("Stop\n\nPress to\nReverse");
        motorState = 2;
        break;
      case 2: // 「正転後のストップ」の状態でボタンが押されると、逆転を開始
        dxl.setGoalCurrent(DXL_ID, -800, UNIT_MILLI_AMPERE);
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.print("Reverse\n\nPress to\nStop");
        motorState = 3;
        break;
      case 3: // 「逆転中」の状態でボタンが押されると、モーターをストップさせ、初期状態に戻る
        dxl.setGoalCurrent(DXL_ID, 0, UNIT_MILLI_AMPERE);
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 0);
        M5.Lcd.print("Stop\n\nPress to\nForward");
        motorState = 0;
        break;
    }
  }
  delay(10);
}
