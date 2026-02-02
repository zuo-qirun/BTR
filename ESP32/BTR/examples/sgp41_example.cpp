#include <Arduino.h>
#include <SensirionCore.h>
#include <SensirionI2CSgp41.h>
#include <Wire.h>

// ESP32-S3 DevKitM-1 I2C pin mapping.
// SGP41 -> ESP32-S3
// SDA   -> GPIO8
// SCL   -> GPIO9
// 3V3   -> 3V3
// GND   -> GND

constexpr uint8_t PIN_I2C_SDA = 8;
constexpr uint8_t PIN_I2C_SCL = 9;

constexpr uint16_t kConditioningSamples = 10;

SensirionI2CSgp41 sgp41;

uint16_t humidityToTicks(float relativeHumidity) {
  if (relativeHumidity < 0.0f) {
    relativeHumidity = 0.0f;
  } else if (relativeHumidity > 100.0f) {
    relativeHumidity = 100.0f;
  }
  return static_cast<uint16_t>(relativeHumidity * 65535.0f / 100.0f + 0.5f);
}

uint16_t temperatureToTicks(float temperatureC) {
  if (temperatureC < -45.0f) {
    temperatureC = -45.0f;
  } else if (temperatureC > 130.0f) {
    temperatureC = 130.0f;
  }
  return static_cast<uint16_t>((temperatureC + 45.0f) * 65535.0f / 175.0f + 0.5f);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println("Sensirion SGP41 Air Quality Example (ESP32-S3)");
  Serial.println("Pin wiring: SDA->GPIO8, SCL->GPIO9, 3V3->3V3, GND->GND");

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  sgp41.begin(Wire);

  uint16_t testResult = 0;
  uint16_t error = sgp41.executeSelfTest(testResult);
  if (error || testResult != 0xD400) {
    char errorMessage[256];
    errorToString(error, errorMessage, sizeof(errorMessage));
    Serial.print("SGP41 self-test failed: ");
    Serial.print(errorMessage);
    Serial.print(" (result=0x");
    Serial.print(testResult, HEX);
    Serial.println(")");
  } else {
    Serial.println("SGP41 self-test passed");
  }
}

void loop() {
  static uint16_t sampleCount = 0;

  uint16_t defaultRh = humidityToTicks(50.0f);
  uint16_t defaultT = temperatureToTicks(25.0f);

  uint16_t srawVoc = 0;
  uint16_t srawNox = 0;
  uint16_t error = 0;

  if (sampleCount < kConditioningSamples) {
    error = sgp41.executeConditioning(defaultRh, defaultT, srawVoc);
    if (error) {
      char errorMessage[256];
      errorToString(error, errorMessage, sizeof(errorMessage));
      Serial.print("SGP41 conditioning failed: ");
      Serial.println(errorMessage);
    } else {
      Serial.print("Conditioning sample ");
      Serial.print(sampleCount + 1);
      Serial.print("/ ");
      Serial.print(kConditioningSamples);
      Serial.print(" SRAW_VOC: ");
      Serial.println(srawVoc);
    }
    sampleCount++;
  } else {
    error = sgp41.measureRawSignals(defaultRh, defaultT, srawVoc, srawNox);
    if (error) {
      char errorMessage[256];
      errorToString(error, errorMessage, sizeof(errorMessage));
      Serial.print("SGP41 read failed: ");
      Serial.println(errorMessage);
    } else {
      Serial.print("SRAW_VOC: ");
      Serial.print(srawVoc);
      Serial.print("  SRAW_NOX: ");
      Serial.println(srawNox);
    }
  }

  delay(1000);
}
