// Mainly copied from dynamixe_current.cpp
#include <Dynamixel2Arduino.h>
#define DXL_SERIAL Serial1
#define SERIAL_CONFIG SERIAL_8N1
const long BAUDRATE = 2000000; // 2Mが限界だった
const int TIMEOUT = 100;  //ms
const uint8_t DXL_ID = 0;
const float DXL_PROTOCOL_VERSION = 2.0;

const int EN_PIN = 6;  // Enable Pin
const int RX_PIN = 5;
const int TX_PIN = 38;

Dynamixel2Arduino dxl(DXL_SERIAL, EN_PIN);

void setupDXL()
{
  DXL_SERIAL.begin(BAUDRATE, SERIAL_CONFIG, RX_PIN, TX_PIN, false, TIMEOUT);
  dxl.begin(BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  dxl.torqueOff(DXL_ID);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT);
  dxl.torqueOn(DXL_ID);

  dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, DXL_ID, 0);
}


// Mainly copied from esp_now_pairing
#define LGFX_M5ATOMS3
#define LGFX_USE_V1
#include <SerialTransfer.h>

#include <LGFX_AUTODETECT.hpp>
#include <LovyanGFX.hpp>

#include "pairing.h"

#ifdef ENV_MAIN
String main_or_secondary = "Main";
#else
String main_or_secondary = "Secondary";
#endif

static LGFX lcd;
Pairing pairing;
String previousMessage = "";
SerialTransfer transfer;
uint8_t buffer[256];

template <typename T>
void write(const T &data, const size_t len) {
    transfer.txObj(data, 0, len);
    transfer.sendData(len);
}

void printToLCD(const String &message) {
    if (message == previousMessage) {
        return;
    }
    previousMessage = message;
    lcd.fillScreen(lcd.color565(0, 0, 0));
    lcd.setCursor(0, 0);
    lcd.println(message);
}

void initLCD() {
    lcd.init();
    lcd.setRotation(0);
    lcd.clear();
    lcd.setTextSize(1.2);
}

void setup() {
    initLCD();
    printToLCD("Initializing...");

    int xCoreID = 1;
    pairing.createTask(xCoreID);
    String displayMessage = String(main_or_secondary) + "\nMy MAC:\n" + pairing.getMyMACAddress();
    printToLCD(displayMessage);

    USBSerial.begin(115200);
    transfer.begin(USBSerial);
    delay(1500);

    if (main_or_secondary == "Main") {
      printToLCD(displayMessage + "\n\nWait for command from espnow_dynamixel_controller.py");
    }
    else if (main_or_secondary == "Secondary") {
      setupDXL();
      printToLCD(displayMessage + "\n\nWait for command from Main");
    }
}

void control_dynamixel(PairingData data) {
  String displayMessage = String(main_or_secondary) + "\nMy MAC:\n" + pairing.getMyMACAddress();
  if (data.IPv4[0] == 0) {
    // dxl.setGoalCurrent(DXL_ID, 10, UNIT_PERCENT);
    dxl.setGoalCurrent(DXL_ID, 500, UNIT_MILLI_AMPERE);
    printToLCD(displayMessage + "\n\nForward rotation");
  }
  else if (data.IPv4[0] == 127) {
    dxl.setGoalCurrent(DXL_ID, 0, UNIT_MILLI_AMPERE);
    printToLCD(displayMessage + "\n\nStop rotation");
  }
  else if (data.IPv4[0] == 255) {
    dxl.setGoalCurrent(DXL_ID, -800, UNIT_MILLI_AMPERE);
    printToLCD(displayMessage + "\n\nReverse rotation");
  }
  else {
    printToLCD(displayMessage + "\n\nUnknown command: " + String(data.IPv4[0]));
  }
}

PairingData dataToSend = {{255, 255, 255, 255}};

void loop() {
    size_t available = transfer.available();
    if (available > 0) {
        transfer.rxObj(buffer, 0, available);
        switch (buffer[0]) {
            case 0x11:
                write(main_or_secondary, main_or_secondary.length());
                break;
            case 0x12: {
                std::map<String, PairingData> pairedDataMap = pairing.getPairedData();
                if (!pairedDataMap.empty()) {
                    auto it = pairedDataMap.begin();
                    write(it->second.IPv4, 4);
                }
                break;
            }
            case 0x13:
                for (int i = 0; i < 4; i++) {
                    dataToSend.IPv4[i] = buffer[i + 1];
                }
                pairing.setDataToSend(dataToSend);
                break;
            default:
                break;
        }
        printToLCD(String(main_or_secondary) + "\nMy MAC:\n" + pairing.getMyMACAddress() + "\n\nReceive command from espnow_dynamixel_controller.py");
    }

    std::map<String, PairingData> pairedDataMap = pairing.getPairedData();
    for (const auto &pair : pairedDataMap) {
        // Additional code to esp_now_pairing
        if (main_or_secondary == "Secondary") {
          control_dynamixel(pair.second);
        }
    }
    delay(100);
}
