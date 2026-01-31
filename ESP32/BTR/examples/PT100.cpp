#include <Arduino.h>
#include <Adafruit_MAX31865.h>
#include <SPI.h>

// ESP32-S3 DevKitM-1 hardware SPI pin mapping.
// MAX31865 -> ESP32-S3
// SCK  -> GPIO12
// MISO -> GPIO13
// MOSI -> GPIO11
// CS   -> GPIO10
// 3V3  -> 3V3
// GND  -> GND

constexpr uint8_t PIN_MAX31865_CS = 10;
constexpr uint8_t PIN_MAX31865_MOSI = 11;
constexpr uint8_t PIN_MAX31865_MISO = 13;
constexpr uint8_t PIN_MAX31865_SCK = 12;

constexpr float RREF = 430.0F; // Use 430.0 for PT100.

Adafruit_MAX31865 max31865(PIN_MAX31865_CS, &SPI);

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println("Adafruit MAX31865 PT100 Sensor Test (ESP32-S3)");

  SPI.begin(PIN_MAX31865_SCK, PIN_MAX31865_MISO, PIN_MAX31865_MOSI, PIN_MAX31865_CS);
  max31865.begin(MAX31865_3WIRE); // Set to MAX31865_2WIRE or MAX31865_4WIRE if needed.
}

void loop() {
  uint16_t rtd = max31865.readRTD();

  Serial.print("RTD value: ");
  Serial.println(rtd);

  float ratio = rtd;
  ratio /= 32768.0F;

  Serial.print("Ratio = ");
  Serial.println(ratio, 8);
  Serial.print("Resistance = ");
  Serial.println(RREF * ratio, 8);
  Serial.print("Temperature = ");
  Serial.println(max31865.temperature(100, RREF));

  uint8_t fault = max31865.readFault();
  if (fault) {
    Serial.print("Fault 0x");
    Serial.println(fault, HEX);
    if (fault & MAX31865_FAULT_HIGHTHRESH) {
      Serial.println("RTD High Threshold");
    }
    if (fault & MAX31865_FAULT_LOWTHRESH) {
      Serial.println("RTD Low Threshold");
    }
    if (fault & MAX31865_FAULT_REFINLOW) {
      Serial.println("REFIN- > 0.85 x Bias");
    }
    if (fault & MAX31865_FAULT_REFINHIGH) {
      Serial.println("REFIN- < 0.85 x Bias - FORCE- open");
    }
    if (fault & MAX31865_FAULT_RTDINLOW) {
      Serial.println("RTDIN- < 0.85 x Bias - FORCE- open");
    }
    if (fault & MAX31865_FAULT_OVUV) {
      Serial.println("Under/Over voltage");
    }
    max31865.clearFault();
  }

  Serial.println();
  delay(1000);
}
