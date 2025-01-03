#include <atom_s3_lcd.h>
#include <robot_2d.h>

// Initialize robot
AtomS3LCD atoms3lcd;
Robot2D robot_2d(atoms3lcd);

void setup() {
}

void loop() {
    // Set Goal
    float goal_x = 0.5f;
    float goal_y = 10.0f;
    float goal_angle = M_PI * 1.25;
    robot_2d.setPose(goal_x, goal_y, goal_angle);
    // Visualize
    int16_t originX = atoms3lcd.width() / 2; // [px]
    int16_t originY = atoms3lcd.height() - 30; // [px]
    float distance = sqrt(goal_x * goal_x + goal_y * goal_y);
    float scale = 50.0f / distance; // 1.0[m] = scale[px]
    float draw_second = 2.0; // [s]
    robot_2d.drawTrajectory(originX, originY, scale, draw_second);
    delay(1000);
}
