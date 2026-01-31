#include <Arduino.h>
#include <FastLED.h>
#define NUM_LEDS 1
#define DATA_PIN 48
CRGB leds[NUM_LEDS];


// put function declarations here:

void setup() {
  // put your setup code here, to run once:
  // pinMode(DATA_PIN, OUTPUT);
  FastLED.addLeds<WS2812, DATA_PIN, GRB>(leds, NUM_LEDS);
  leds[0] = CRGB::Red;
  FastLED.show();
}

void loop() {
  // put your main code here, to run repeatedly:
  
}

// put function definitions here:
