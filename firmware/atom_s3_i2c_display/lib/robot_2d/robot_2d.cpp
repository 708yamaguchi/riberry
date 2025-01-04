#include <robot_2d.h>

Robot2D* Robot2D::instance = nullptr;

Robot2D::Robot2D(AtomS3LCD &lcd)
  : atoms3lcd(lcd) {
    instance = this;
}

void Robot2D::setPose(float x, float y, float angle) {
  x_ = x;
  y_ = y;
  angle_ = angle;
}

float Robot2D::getX() {
  return x_;
}

float Robot2D::getY() {
  return y_;
}

float Robot2D::getAngle() {
  return angle_;
}

void Robot2D::draw(int16_t originX, int16_t originY, float scale) const {
  // Home-base pentagon shape definition
  int16_t baseSize = 30; // Size of the robot in pixels
  // Pentagon vertices
  int16_t x1 = 0,             y1 = -baseSize / 2.5;  // Top
  int16_t x2 = -baseSize / 2, y2 = 0;              // Top left
  int16_t x3 = baseSize / 2,  y3 = 0;              // Top right
  int16_t x4 = -baseSize / 2, y4 = baseSize / 1.5; // Bottom left
  int16_t x5 = baseSize / 2,  y5 = baseSize / 1.5; // Bottom right

  // Rotation processing
  float cosTheta = cos(angle_);
  float sinTheta = sin(angle_);

  // Rotate vertices
  int16_t rx1 = cosTheta * x1 - sinTheta * y1, ry1 = sinTheta * x1 + cosTheta * y1;
  int16_t rx2 = cosTheta * x2 - sinTheta * y2, ry2 = sinTheta * x2 + cosTheta * y2;
  int16_t rx3 = cosTheta * x3 - sinTheta * y3, ry3 = sinTheta * x3 + cosTheta * y3;
  int16_t rx4 = cosTheta * x4 - sinTheta * y4, ry4 = sinTheta * x4 + cosTheta * y4;
  int16_t rx5 = cosTheta * x5 - sinTheta * y5, ry5 = sinTheta * x5 + cosTheta * y5;

  // Screen coordinates
  int16_t screenX = originX + static_cast<int16_t>(x_ * scale);
  int16_t screenY = originY - static_cast<int16_t>(y_ * scale);

  // Draw pentagon using triangles
  instance->atoms3lcd.fillTriangle(screenX + rx1, screenY + ry1, screenX + rx2, screenY + ry2, screenX + rx3, screenY + ry3, TFT_GREEN);
  instance->atoms3lcd.fillTriangle(screenX + rx2, screenY + ry2, screenX + rx3, screenY + ry3, screenX + rx4, screenY + ry4, TFT_GREEN);
  instance->atoms3lcd.fillTriangle(screenX + rx3, screenY + ry3, screenX + rx4, screenY + ry4, screenX + rx5, screenY + ry5, TFT_GREEN);
}

/**
 * @brief Draws the distance traveled on the specified LGFX display.
 */
void Robot2D::drawDistance(float distance, int16_t x, int16_t y) const {
  int32_t origCursorX = instance->atoms3lcd.getCursorX();
  int32_t origCursorY = instance->atoms3lcd.getCursorY();
  instance->atoms3lcd.setCursor(x, y);
  float origTextSize = instance->atoms3lcd.getTextSize();
  instance->atoms3lcd.setTextSize(1.5);
  instance->atoms3lcd.setTextColor(TFT_WHITE, TFT_BLACK);
  instance->atoms3lcd.printColorText(String("Dist ") + String(distance) + String("[m]"));
  instance->atoms3lcd.setTextSize(origTextSize);
  instance->atoms3lcd.setCursor(origCursorX, origCursorY);
}

/**
 * @param Initial scale. Scale factor to convert meters to pixels.
 */
void Robot2D::drawTrajectory(int16_t originX, int16_t originY, float scale, float draw_second) {
  float distance = sqrt(getX() * getX() + getY() * getY());
  float orig_x = getX(), orig_y = getY(), orig_angle = getAngle();
  // Initialize robot pose
  setPose(0, 0, 0);
  // Draw
  float draw_interval = 0.05; // [s]
  int div = (int)(draw_second / draw_interval);
  for (int i = 0; i < div; i++) {
    setPose(getX() + orig_x / div,
            getY() + orig_y / div,
            getAngle() + orig_angle / div);
    instance->atoms3lcd.fillScreen(TFT_BLACK);
    draw(originX, originY, scale);
    drawDistance(distance, 10, 10);
    vTaskDelay(pdMS_TO_TICKS(draw_interval * 1000));
  }
  // Reset robot pose
  setPose(orig_x, orig_y, orig_angle);
}
