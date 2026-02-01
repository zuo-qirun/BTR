#include <Arduino.h>

// MQ-2 gas concentration example (ESP32-S3)
// Pin connections:
//   MQ-2 VCC  -> 5V (or 3.3V module if supported)
//   MQ-2 GND  -> GND
//   MQ-2 AO   -> GPIO4 (ADC1_CH3) through a voltage divider if AO can exceed 3.3V
//   MQ-2 DO   -> (unused in this example)
// Notes:
// - MQ-2 heater needs warm-up time (several minutes) for stable readings.
// - AO can be up to 5V on many modules; use a divider to protect ESP32 ADC.
// - ppm conversion here is a rough placeholder. Calibrate R0 for your sensor.

constexpr int kMq2AnalogPin = 4;
constexpr uint32_t kBaudRate = 115200;
constexpr float kAdcMax = 4095.0f;
constexpr float kVref = 3.3f;

// Calibrated clean-air resistance (replace after calibration).
constexpr float kR0 = 10.0f; // kΩ

float EstimatePpmFromRatio(float ratio) {
  // Placeholder curve (log-log). Adjust for target gas and calibration.
  // Example for LPG-like response: ppm ≈ 10 ^ ((log10(ratio) - b) / m)
  const float m = -0.47f;
  const float b = 1.83f;
  return powf(10.0f, (log10f(ratio) - b) / m);
}

void setup() {
  Serial.begin(kBaudRate);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Full-scale ~3.3V

  Serial.println("MQ-2 gas concentration example");
  Serial.println("Connections: VCC->5V, GND->GND, AO->GPIO4 (with divider if needed)");
}

void loop() {
  int raw = analogRead(kMq2AnalogPin);
  float voltage = (raw / kAdcMax) * kVref;

  // Simple Rs calculation for a load resistor of 10kΩ.
  const float kRl = 10.0f; // kΩ
  float rs = (kVref - voltage) * kRl / max(voltage, 0.001f);
  float ratio = rs / kR0;
  float ppm = EstimatePpmFromRatio(ratio);

  Serial.print("ADC: ");
  Serial.print(raw);
  Serial.print("\tVoltage: ");
  Serial.print(voltage, 3);
  Serial.print(" V");
  Serial.print("\tRs/R0: ");
  Serial.print(ratio, 2);
  Serial.print("\tGas (est.): ");
  Serial.print(ppm, 1);
  Serial.println(" ppm");

  delay(1000);
}
