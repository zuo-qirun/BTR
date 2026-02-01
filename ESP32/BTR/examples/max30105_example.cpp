#include <Arduino.h>
#include <Wire.h>
#include <MAX30105.h>

// MAX30105 smoke concentration example (ESP32-S3)
// Pin connections:
//   MAX30105 VIN  -> 3.3V
//   MAX30105 GND  -> GND
//   MAX30105 SDA  -> GPIO8 (I2C SDA)
//   MAX30105 SCL  -> GPIO9 (I2C SCL)
//   MAX30105 INT  -> (optional) GPIO7
// Notes:
// - This example uses the IR channel as a proxy for smoke/particle density.
// - The conversion to "smoke concentration" is a simple linear mapping and
//   should be calibrated for your enclosure and sensor setup.

constexpr int kI2cSda = 8;
constexpr int kI2cScl = 9;
constexpr uint32_t kBaudRate = 115200;

MAX30105 sensor;

float MapIrToSmokePpm(uint32_t irValue) {
  // Simple placeholder mapping. Adjust with calibration data.
  // Clamp to avoid overflow and keep output readable.
  const float kScale = 0.0025f;
  return min(irValue * kScale, 1000.0f);
}

void setup() {
  Serial.begin(kBaudRate);
  Wire.begin(kI2cSda, kI2cScl);
  Wire.setClock(400000);

  Serial.println("MAX30105 smoke concentration example");
  Serial.println("Connections: VIN->3.3V, GND->GND, SDA->GPIO8, SCL->GPIO9, INT->GPIO7 (optional)");

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30105 not detected. Check wiring.");
    while (true) {
      delay(1000);
    }
  }

  sensor.setup();
  sensor.setPulseAmplitudeRed(0x00);   // Disable red LED
  sensor.setPulseAmplitudeGreen(0x00); // Disable green LED
  sensor.setPulseAmplitudeIR(0x1F);    // IR LED at low power
}

void loop() {
  uint32_t irValue = sensor.getIR();
  float smokePpm = MapIrToSmokePpm(irValue);

  Serial.print("IR Raw: ");
  Serial.print(irValue);
  Serial.print("\tSmoke (est.): ");
  Serial.print(smokePpm, 1);
  Serial.println(" ppm");

  delay(500);
}
