#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <SensirionI2CSht4x.h>
#include <SensirionI2CSgp41.h>
#include <Wire.h>

namespace {
constexpr uint8_t kButtonPin = 17;
constexpr uint8_t kScreenWidth = 128;
constexpr uint8_t kScreenHeight = 64;
constexpr uint8_t kOledReset = 255; // no reset pin
constexpr uint8_t kSamplesPerMinute = 60;
constexpr uint32_t kSampleIntervalMs = 1000;
constexpr uint32_t kGraphDurationMs = 60000;

Adafruit_SSD1306 display(kScreenWidth, kScreenHeight, &Wire, kOledReset);
SensirionI2CSht4x sht4x;
SensirionI2CSgp41 sgp41;

float tempHistory[kSamplesPerMinute] = {};
float humidityHistory[kSamplesPerMinute] = {};
uint8_t historyIndex = 0;
uint8_t historyCount = 0;
bool buttonLatched = false;
uint32_t lastSampleMs = 0;
uint32_t graphStartMs = 0;
} // namespace

void UpdateHistory(float temperature, float humidity) {
  tempHistory[historyIndex] = temperature;
  humidityHistory[historyIndex] = humidity;
  historyIndex = (historyIndex + 1) % kSamplesPerMinute;
  if (historyCount < kSamplesPerMinute) {
    historyCount++;
  }
}

void DrawLiveScreen(float temperature, float humidity, uint16_t vocRaw) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("SHT4x + SGP41");
  display.setTextSize(2);
  display.setCursor(0, 16);
  display.print("T ");
  display.print(temperature, 1);
  display.println("C");
  display.setCursor(0, 36);
  display.print("H ");
  display.print(humidity, 1);
  display.println("%");
  display.setTextSize(1);
  display.setCursor(0, 56);
  display.print("VOC raw ");
  display.print(vocRaw);
  display.display();
}

void DrawGraphScreen() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Last 60s");

  if (historyCount == 0) {
    display.setCursor(0, 20);
    display.println("No data");
    display.display();
    return;
  }

  float tempMin = tempHistory[0];
  float tempMax = tempHistory[0];
  float humMin = humidityHistory[0];
  float humMax = humidityHistory[0];
  for (uint8_t i = 1; i < historyCount; ++i) {
    tempMin = min(tempMin, tempHistory[i]);
    tempMax = max(tempMax, tempHistory[i]);
    humMin = min(humMin, humidityHistory[i]);
    humMax = max(humMax, humidityHistory[i]);
  }
  if (tempMax - tempMin < 0.5f) {
    tempMax = tempMin + 0.5f;
  }
  if (humMax - humMin < 1.0f) {
    humMax = humMin + 1.0f;
  }

  const int graphTop = 12;
  const int graphBottom = 63;
  const int graphHeight = graphBottom - graphTop;
  const int graphWidth = 127;

  for (uint8_t i = 0; i < historyCount; ++i) {
    uint8_t index = (historyIndex + kSamplesPerMinute - historyCount + i) % kSamplesPerMinute;
    float tempValue = tempHistory[index];
    float humValue = humidityHistory[index];
    int x = map(i, 0, kSamplesPerMinute - 1, 0, graphWidth);
    int tempY = graphBottom - (int)((tempValue - tempMin) / (tempMax - tempMin) * graphHeight);
    int humY = graphBottom - (int)((humValue - humMin) / (humMax - humMin) * graphHeight);
    display.drawPixel(x, tempY, SSD1306_WHITE);
    display.drawPixel(x, humY, SSD1306_WHITE);
  }

  display.setTextSize(1);
  display.setCursor(0, 56);
  display.print("T:");
  display.print(tempMin, 1);
  display.print("-");
  display.print(tempMax, 1);
  display.setCursor(64, 56);
  display.print("H:");
  display.print(humMin, 0);
  display.print("-");
  display.print(humMax, 0);
  display.display();
}

void setup() {
  Serial.begin(115200);
  pinMode(kButtonPin, INPUT_PULLUP);
  Wire.begin();

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 init failed");
    while (true) {
      delay(1000);
    }
  }
  display.clearDisplay();
  display.display();

  sht4x.begin(Wire);
  sgp41.begin(Wire);

  uint16_t error = sht4x.stopPeriodicMeasurement();
  if (error) {
    Serial.print("SHT4x stop periodic error: ");
    Serial.println(error);
  }
  delay(500);
  error = sht4x.startPeriodicMeasurement();
  if (error) {
    Serial.print("SHT4x start periodic error: ");
    Serial.println(error);
  }

  uint16_t vocIndex = 0;
  uint16_t noxIndex = 0;
  error = sgp41.executeConditioning(vocIndex, noxIndex);
  if (error) {
    Serial.print("SGP41 conditioning error: ");
    Serial.println(error);
  }
}

void loop() {
  const uint32_t now = millis();
  if (now - lastSampleMs >= kSampleIntervalMs) {
    lastSampleMs = now;

    float temperature = 0.0f;
    float humidity = 0.0f;
    uint16_t error = sht4x.readMeasurement(temperature, humidity);
    if (error) {
      Serial.print("SHT4x read error: ");
      Serial.println(error);
    }

    uint16_t vocRaw = 0;
    uint16_t noxIndex = 0;
    error = sgp41.measureRawSignals(0, 0, vocRaw, noxIndex);
    if (error) {
      Serial.print("SGP41 measure error: ");
      Serial.println(error);
    }

    UpdateHistory(temperature, humidity);

    if (graphStartMs != 0 && now - graphStartMs < kGraphDurationMs) {
      DrawGraphScreen();
    } else {
      graphStartMs = 0;
      DrawLiveScreen(temperature, humidity, vocRaw);
    }
  }

  bool buttonPressed = digitalRead(kButtonPin) == LOW;
  if (buttonPressed && !buttonLatched) {
    buttonLatched = true;
    graphStartMs = now;
    DrawGraphScreen();
  } else if (!buttonPressed) {
    buttonLatched = false;
  }
}
