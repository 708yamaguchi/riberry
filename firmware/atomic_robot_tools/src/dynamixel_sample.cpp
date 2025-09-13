#include <Arduino.h>
#include <Dynamixel2Arduino.h>
#include <M5AtomS3.h>


// Dynamixel2Arduino ////////////////////////////////////////////////////////
//This namespace is required to use Control table item names
using namespace ControlTableItem;

// Serial:
// 8 bit
// 1 Stop bit
// None Parity
// 57600 baud (Default)
// ID: 1 (Default)
#define DXL_SERIAL Serial1
#define SERIAL_CONFIG SERIAL_8N1
const long BAUDRATE = 57600;
const int TIMEOUT = 100;  //ms
const uint8_t DXL_ID = 1;
const float DXL_PROTOCOL_VERSION = 2.0;

// Pin
const int DXL_DIR_PIN = 6;  // Enable Pin
const int RX_PIN = 5;
const int TX_PIN = 38;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
/////////////////////////////////////////////////////////////////////////////


void setup() {
  // put your setup code here, to run once:
  M5.begin();
  M5.Lcd.setRotation(0);  // 画面向き設定（USB位置基準 0：下/ 1：右/ 2：上/ 3：左）
  M5.Lcd.setTextSize(2);  // 文字サイズ（整数倍率）
  M5.Lcd.print("Dynamixel Sample");

  // Serial.begin()を呼ばないと、dxl.begin()が終わらない
  // dxl.beginの引数でbaudを指定するため、ここのBAUDRATEは何でも良い
  Serial1.begin(BAUDRATE, SERIAL_CONFIG, RX_PIN, TX_PIN, false, TIMEOUT);

  // Set Port baudrate to 57600bps. This has to match with DYNAMIXEL baudrate.
  dxl.begin(BAUDRATE);
  // Set Port Protocol Version. This has to match with DYNAMIXEL protocol version.
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  // Get DYNAMIXEL information
  dxl.ping(DXL_ID);

  // Turn off torque when configuring items in EEPROM area
  dxl.torqueOff(DXL_ID);
  dxl.setOperatingMode(DXL_ID, OP_POSITION);
  dxl.torqueOn(DXL_ID);

  // Limit the maximum velocity in Position Control Mode. Use 0 for Max speed
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 0);

  M5.Lcd.clear();
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.print("ID: ");
  M5.Lcd.print(DXL_ID);

  delay(2000);
}


void loop() {
  // put your main code here, to run repeatedly:
  
  // Please refer to e-Manual(http://emanual.robotis.com/docs/en/parts/interface/dynamixel_shield/) for available range of value. 
  // Set Goal Position in RAW value
  dxl.setGoalPosition(DXL_ID, 1000);

  int i_present_position = 0;
  float f_present_position = 0.0;

  while (abs(1000 - i_present_position) > 10)
  {
    i_present_position = dxl.getPresentPosition(DXL_ID);

    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print(i_present_position);
  }
  delay(1000);

  // Set Goal Position in DEGREE value
  dxl.setGoalPosition(DXL_ID, 5.7, UNIT_DEGREE);
  
  while (abs(5.7 - f_present_position) > 2.0)
  {
    f_present_position = dxl.getPresentPosition(DXL_ID, UNIT_DEGREE);
    
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print(f_present_position);
  }
  delay(1000);
}