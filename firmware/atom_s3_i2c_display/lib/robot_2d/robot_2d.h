#ifndef ROBOT_2D_H
#define ROBOT_2D_H

#include <cmath>
#include <atom_s3_lcd.h>

/**
 * @class Robot2D
 * @brief Represents a 2D robot that can be drawn on a graphical display.
 *
 * This class encapsulates the properties and methods for drawing a 2D robot with a home-base shape (pentagon)
 * and displaying the distance traveled on a graphical display using the LGFX library.
 */
class Robot2D {
public:
    /**
     * @brief Constructor to initialize the Robot2D with default or specified values.
     */
    Robot2D(AtomS3LCD &lcd);
    /**
     * @param x Initial x-coordinate in meters (default: 0.0f).
     * @param y Initial y-coordinate in meters (default: 0.0f).
     * @param angle Initial orientation in radians (default: 0.0f).
     */
    void setPose(float x, float y, float angle);
    float getX();
    float getY();
    float getAngle();

    /**
     * @brief Draws the robot on the graphical display.
     * @param originX X-coordinate of the origin in pixels.
     * @param originY Y-coordinate of the origin in pixels.
     *
     * This function calculates the robot's shape and orientation, then renders it on the display
     * along with the total distance traveled.
     */

    void draw(int16_t originX, int16_t originY, float scale) const;

    /**
     * @brief Draws the distance traveled on the specified LGFX display.
     */
    void drawDistance(float distance, int16_t x, int16_t y) const;

    /**
     * @param Initial scale. Scale factor to convert meters to pixels.
     * @param originX X-coordinate of the origin in pixels.
     * @param originY Y-coordinate of the origin in pixels.
     */
    void drawTrajectory(int16_t originX, int16_t originY, float scale, float draw_second);

private:
    static Robot2D* instance; /**< Singleton instance of Robot2D. */
    AtomS3LCD &atoms3lcd;
    float x_ = 0;     ///< Position in meters.
    float y_ = 0;     ///< Position in meters.
    float angle_ = 0; ///< Orientation in radians.
};

#endif // ROBOT_2D_H
