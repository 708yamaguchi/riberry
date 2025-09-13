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
const float DXL_PROTOCOL_VERSION = 2.0;

// Pin
const int DXL_DIR_PIN = 6;  // Enable Pin
const int RX_PIN = 5;
const int TX_PIN = 38;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

uint8_t dxl_id = 1;
uint8_t new_dxl_id = 0;
bool scanID(uint8_t* ID_found)
{
    for (uint8_t id = 0; id < 253; id++)
    {
        if (dxl.ping(id))
        {
            *ID_found = id;
            return true;
        }
    }
    return false;
}
/////////////////////////////////////////////////////////////////////////////


void setup() {
  M5.begin();
  M5.Lcd.setRotation(0);  // 画面向き設定（USB位置基準 0：下/ 1：右/ 2：上/ 3：左）
  M5.Lcd.setTextSize(2);  // 文字サイズ（整数倍率）
  M5.Lcd.print("Dynamixel Set ID");

  // Initialize dxl
  DXL_SERIAL.begin(BAUDRATE, SERIAL_CONFIG, RX_PIN, TX_PIN, false, TIMEOUT);
  dxl.begin(BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  // 接続されているDynamixelのIDを検索
  bool found_id = scanID(&dxl_id);
  if (found_id)
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print("Found ID: ");
    M5.Lcd.print(dxl_id);
  }
  else
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print("ID was not found");
    return;
  }

  delay(2000);

  M5.Lcd.print("\n");
  M5.Lcd.print("Press button to set ID to: ");
  M5.Lcd.print(new_dxl_id);
  do { M5.update(); delay(10); } while (!M5.BtnA.isPressed());

  bool set_id_succeed = dxl.setID(dxl_id, new_dxl_id);
  if (set_id_succeed)
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print("Succeed setting to new ID: ");
    M5.Lcd.print(new_dxl_id);
  }
  else
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print("Failed setting to new ID");
  }
}


void loop() {
}
