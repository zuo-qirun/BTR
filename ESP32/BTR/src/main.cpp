#include <Arduino.h>
#include <FastLED.h>
#define NUM_LEDS 1
#define DATA_PIN 48
CRGB leds[NUM_LEDS];


// put function declarations here:

void setup() {
  // put your setup code here, to run once:
  FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);
}

void loop() {
  // put your main code here, to run repeatedly:
  leds[0] = CRGB::Red;
  FastLED.show();
  delay(1000);
  leds[0] = CRGB::Green;
  FastLED.show();
  delay(1000);
  leds[0] = CRGB::Blue;
  FastLED.show();
  delay(1000);
}

// put function definitions here:
