#include <teaching_mode.h>
#include <robot_2d.h>

TeachingMode* TeachingMode::instance = nullptr;
String TeachingMode::prevColorStr = ""; // Initialize the static prevColorStr

TeachingMode::TeachingMode(AtomS3LCD &lcd, AtomS3I2C &i2c)
  : atoms3lcd(lcd), atoms3i2c(i2c), Mode("TeachingMode") {
    instance = this;
}

void TeachingMode::task(void *parameter) {
  while (true) {
    instance->atoms3i2c.setRequestStr(instance->getModeName());
    // Check for I2C timeout
    if (instance->atoms3i2c.checkTimeout()) {
      instance->atoms3lcd.drawNoDataReceived();
      instance->atoms3lcd.printColorText(instance->getModeName() + "\n");
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }
    // Display information
    else {
      if (instance->atoms3lcd.color_str.isEmpty()) {
        instance->atoms3lcd.drawBlack();
        instance->atoms3lcd.printColorText("Waiting for " + instance->getModeName());
        vTaskDelay(pdMS_TO_TICKS(500));
      }
      else {
        // Do nothing if atoms3lcd.color_str is not changed
        if (instance->prevColorStr.equals(instance->atoms3lcd.color_str)) {
          vTaskDelay(pdMS_TO_TICKS(50));
          continue;
        }
        else {
          instance->prevColorStr = instance->atoms3lcd.color_str;
          instance->atoms3lcd.drawBlack();
        }
        int listSize = 5;
        // Draw string
        char** StrList = (char**)malloc(listSize * sizeof(char*));
        int modeCount = instance->atoms3i2c.splitString(instance->atoms3lcd.color_str, ',', StrList, listSize);
        instance->atoms3lcd.printColorText(String(StrList[0]));
        // Draw AR Marker if found
        if (!String(StrList[1]).equals(String(""))) {
          int marker_id = atoi(StrList[1]);
          int width = instance->atoms3lcd.width();
          int height = instance->atoms3lcd.height();
          int size = 64;
          instance->drawARMarker(marker_id, width - size - 5, height - size - 5, size);
        }
        // Draw pose correction if specified
        if (!String(StrList[2]).equals(String("")) &&
            !String(StrList[3]).equals(String("")) &&
            !String(StrList[4]).equals(String(""))) {
          Robot2D robot_2d = Robot2D(instance->atoms3lcd);
          float goal_x = (float)atof(StrList[2]);
          float goal_y = (float)atof(StrList[3]);
          float goal_angle = (float)atof(StrList[4]);
          robot_2d.setPose(goal_x, goal_y, goal_angle);
          int16_t originX = instance->atoms3lcd.width() / 2; // [px]
          int16_t originY = instance->atoms3lcd.height() - 20; // [px]
          float distance = sqrt(goal_x * goal_x + goal_y * goal_y); // [m]
          float scale = 50.0f / distance; // 1.0[m] = scale[px]
          float draw_second = 1.5; // [s]
          for (int i=0; i<2; i++) {
            robot_2d.drawTrajectory(originX, originY, scale, draw_second);
            vTaskDelay(pdMS_TO_TICKS(500));
          }
        }
      }
    }
  }
}

// 引数: marker_id: マーカーのID, x: 左上のX座標, y: 左上のY座標, size: マーカーのサイズ
void TeachingMode::drawARMarker(int marker_id, int x, int y, int size) {
  if (size < 8) {
    Serial.println("Error: Marker size must be at least 8.");
    return;
  }

  // 黒背景を描画
  instance->atoms3lcd.fillRect(x, y, size, size, TFT_BLACK);
  // 白い外枠を描画
  instance->atoms3lcd.fillRect(x + size / 8, y + size / 8, size * 6 / 8, size * 6 / 8, TFT_WHITE);
  // 内部の黒枠を描画
  instance->atoms3lcd.fillRect(x + size * 2 / 8, y + size * 2 / 8, size * 4 / 8, size * 4 / 8, TFT_BLACK);
  // ARマーカーの模様を描画
  // 模様を簡易的に固定パターンとして定義
  int pattern[4][4] = {
                       {1, 0, 0, 1},
                       {0, 1, 1, 0},
                       {1, 1, 0, 0},
                       {0, 0, 1, 1}
  };

  int cellSize = size / 8;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (pattern[row][col] == 1) {
        instance->atoms3lcd.fillRect(x + (col + 2) * cellSize, y + (row + 2) * cellSize, cellSize, cellSize, TFT_WHITE);
      }
    }
  }

  if (marker_id >= 0) {
    float textSizeOrig = instance->atoms3lcd.getTextSize();
    float text_size = size / 32.0;
    // 黒い円で内部を抜く
    instance->atoms3lcd.fillCircle(x + size / 2, y + size / 2, text_size * 5.0, TFT_BLACK);
    // マーカーIDを描画
    instance->atoms3lcd.setTextSize(text_size);
    String color_id = String("\x1b[32m") + String(marker_id) + String("\x1b[39m");
    instance->atoms3lcd.setCursor(x + size / 2 - instance->atoms3lcd.textWidth(String(marker_id)) / 2,
                                  y + size / 2 - instance->atoms3lcd.textWidth(String("a")) / 2);
    instance->atoms3lcd.printColorText(color_id);
    // Restore text settings
    instance->atoms3lcd.setTextSize(textSizeOrig);
    instance->atoms3lcd.setCursor(0, 0);
  }
}

void TeachingMode::createTask(uint8_t xCoreID) {
  xTaskCreatePinnedToCore(task, "Teaching Mode", 2048, NULL, 1, &taskHandle, xCoreID);
}
