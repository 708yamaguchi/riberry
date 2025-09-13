#include <Arduino.h>
#include <Dynamixel2Arduino.h>
#include <M5AtomS3.h>


// Dynamixel2Arduino ////////////////////////////////////////////////////////
using namespace ControlTableItem;

#define DXL_SERIAL Serial1
#define SERIAL_CONFIG SERIAL_8N1
const int TIMEOUT = 100;  //ms
const float DXL_PROTOCOL_VERSION = 2.0;

const int DXL_DIR_PIN = 6;  // Enable Pin
const int RX_PIN = 5;
const int TX_PIN = 38;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

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
 
// 指定可能なBaudrateはemanual(https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/)参照
const long baudrates[] = {
  9600,
  57600,
  115200,
  1000000,
  2000000,
  3000000,
  4000000,
  };

// DynamixelのIDとBaudrateを取得
bool scanDXL(uint8_t* ID_found, long* baud_found)
{
  for (long baudrate : baudrates)
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print(baudrate);

    // DXL_SERIAL.end();
    DXL_SERIAL.begin(baudrate);
    dxl.begin(baudrate);
    dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
    for (uint8_t id = 0; id < 253; id++)
    {
      if (dxl.ping(id))
      {
        *ID_found = id;
        *baud_found = baudrate;
        return true;
      }
    }
    delay(100);
  }
  return false;
}
/////////////////////////////////////////////////////////////////////////////


const long NEW_BAUDRATE = 4000000;


void setup() {
  M5.begin();
  M5.Lcd.setRotation(0);  // 画面向き設定（USB位置基準 0：下/ 1：右/ 2：上/ 3：左）
  M5.Lcd.setTextSize(2);  // 文字サイズ（整数倍率）
  M5.Lcd.print("Dynamixel Set Baudrate");

  // Initialize dxl
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  uint8_t dxl_id;
  long dxl_baud;
  bool scan_succeed = scanDXL(&dxl_id, &dxl_baud);
  if(scan_succeed)
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print("Scan succeed");
    M5.Lcd.print("\n");
    M5.Lcd.print("ID: ");
    M5.Lcd.print(dxl_id);
    M5.Lcd.print("Baud: ");
    M5.Lcd.print(dxl_baud);
  }
  else
  {
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.print("Scan failed");
    delay(1000);
    ESP.restart();
  }

  delay(2000);


  M5.Lcd.print("Press button to set baud to: ");
  M5.Lcd.print(NEW_BAUDRATE);
  do { M5.update(); delay(10); } while (!M5.BtnA.isPressed());


  dxl.torqueOff(dxl_id);
  bool set_baud_succeed = dxl.setBaudrate(dxl_id, NEW_BAUDRATE);
  M5.Lcd.clear();
  M5.Lcd.setCursor(0, 0);
  if (set_baud_succeed)
  {
    M5.Lcd.print("Succeed");
    dxl.begin(NEW_BAUDRATE);
  }
  else
    M5.Lcd.print("Failed");
}


void loop() {
}
