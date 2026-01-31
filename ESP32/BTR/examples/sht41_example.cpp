#include <Arduino.h>
#include <SensirionCore.h>
#include <SensirionI2CSht4x.h>
#include <Wire.h>

// ESP32-S3 DevKitM-1 I2C pin mapping.
// SHT40 -> ESP32-S3
// SDA   -> GPIO8
// SCL   -> GPIO9
// 3V3   -> 3V3
// GND   -> GND

constexpr uint8_t PIN_I2C_SDA = 8;
constexpr uint8_t PIN_I2C_SCL = 9;

SensirionI2CSht4x sht4x;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println("Sensirion SHT40 Temperature/Humidity Example (ESP32-S3)");

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  sht4x.begin(Wire);

  uint16_t error = sht4x.softReset();
  if (error) {
    char errorMessage[256];
    errorToString(error, errorMessage, sizeof(errorMessage));
    Serial.print("SHT40 soft reset failed: ");
    Serial.println(errorMessage);
  }
}

void loop() {
  float temperature = 0.0f;
  float humidity = 0.0f;
  uint16_t error = sht4x.measureHighPrecision(temperature, humidity);

  if (error) {
    char errorMessage[256];
    errorToString(error, errorMessage, sizeof(errorMessage));
    Serial.print("SHT40 read failed: ");
    Serial.println(errorMessage);
  } else {
    Serial.print("Temperature: ");
    Serial.print(temperature, 2);
    Serial.print(" °C, Humidity: ");
    Serial.print(humidity, 2);
    Serial.println(" %RH");
  }

  delay(1000);
}
