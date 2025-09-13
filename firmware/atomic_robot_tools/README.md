# Atomic Robot Tools

## Control Dynamixel motor via AtomS3 monitor clicks

1. Burn firmware to device

```
pio run -t upload -e dynamixel_current
```

## Control Dynamixel motor via ESPNOW communicaation

1. Burn firmware to pairing devices

```
# For main (sender) device.
# This device is usually connected to ROS master.
pio run -t upload -e espnow_dynamixel_controller_main

# For receiver device.
# This secondary (device) is usually connected to atomic tools.
pio run -t upload -e espnow_dynamixel_controller_secondary
```

2. Start the pairing program on the computer connected to the main (sender) device

```
rosrun riberry_startup espnow_dynamixel_controller.py
```
