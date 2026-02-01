#include <Arduino.h>

// 49E Hall sensor (linear analog) example for ESP32
// Pin connections:
//   49E VCC -> 3.3V
//   49E GND -> GND
//   49E OUT -> GPIO8 (ADC1_CH6)
// Note: GPIO8 is input-only, suitable for analog read.

constexpr int kHallPin = 8; // ADC1_CH2
constexpr uint32_t kBaudRate = 115200;

void setup() {
  Serial.begin(kBaudRate);
  analogReadResolution(12); // ESP32 default 12-bit
  analogSetAttenuation(ADC_11db); // full-scale ~3.3V

  Serial.println("49E Hall sensor analog output example");
  Serial.println("Connections: VCC->3.3V, GND->GND, OUT->GPIO8");
}

void loop() {
  int raw = analogRead(kHallPin);
  float voltage = (raw / 4095.0f) * 3.3f;

  Serial.print("Raw ADC: ");
  Serial.print(raw);
  Serial.print("\tVoltage: ");
  Serial.print(voltage, 3);
  Serial.println(" V");

  delay(500);
}
